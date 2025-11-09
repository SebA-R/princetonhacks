from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, List

import librosa
import numpy as np
import soundfile as sf
from audiomentations import (
    AddGaussianNoise,
    ClippingDistortion,
    Compose,
    Gain,
    PitchShift,
    Shift,
)
from noisereduce import reduce_noise

from .config import CONFIG, TrainingConfig


def _build_augmentation_pipeline(sample_rate: int) -> Compose:
    return Compose(
        [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            Gain(min_gain_db=-6, max_gain_db=6, p=0.4),
            PitchShift(
                min_semitones=-2,
                max_semitones=2,
                p=0.3,
            ),
            Shift(min_shift=-0.2, max_shift=0.2, shift_unit="fraction", p=0.3, rollover=True),
            ClippingDistortion(
                min_percentile_threshold=0,
                max_percentile_threshold=20,
                p=0.2,
            ),
        ],
        p=0.9,
        shuffle=False,
    )


AUG_PIPELINE = _build_augmentation_pipeline(CONFIG.sample_rate)


def load_audio(
    path: str,
    target_sr: int = CONFIG.sample_rate,
) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32), sr


def trim_silence(audio: np.ndarray, top_db: float) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed if trimmed.size else audio


def denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    return reduce_noise(y=audio, sr=sr, stationary=False)


def loudness_normalize(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    rms = math.sqrt(float(np.mean(audio**2)) + 1e-12)
    current_dbfs = 20 * math.log10(rms + 1e-12)
    gain_db = target_dbfs - current_dbfs
    factor = 10 ** (gain_db / 20)
    normalized = audio * factor
    max_val = np.max(np.abs(normalized)) + 1e-12
    if max_val > 1.0:
        normalized = normalized / max_val
    return normalized.astype(np.float32)


def clip_or_pad(
    audio: np.ndarray,
    sr: int,
    min_duration: float,
    max_duration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    min_len = int(min_duration * sr)
    max_len = int(max_duration * sr)
    if audio.size < min_len:
        pad_amount = min_len - audio.size
        audio = np.pad(audio, (0, pad_amount), mode="reflect")
    if audio.size > max_len:
        max_start = audio.size - max_len
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        audio = audio[start : start + max_len]
    return audio.astype(np.float32)


def preprocess_audio(
    path: str,
    config: TrainingConfig = CONFIG,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(config.random_seed)
    audio, sr = load_audio(path, target_sr=config.sample_rate)
    audio = trim_silence(audio, config.top_db_trim)
    if audio.size == 0:
        audio = np.zeros(int(config.min_duration_sec * config.sample_rate), dtype=np.float32)
    audio = denoise(audio, sr)
    audio = loudness_normalize(audio, config.target_dbfs)
    audio = clip_or_pad(audio, sr, config.min_duration_sec, config.max_duration_sec, rng)
    return audio.astype(np.float32)


def augment_waveform(
    audio: np.ndarray,
    sample_rate: int = CONFIG.sample_rate,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(CONFIG.random_seed)
    AUG_PIPELINE.random_state = np.random.RandomState(int(rng.integers(0, 1_000_000)))
    augmented = AUG_PIPELINE(samples=audio, sample_rate=sample_rate)
    return augmented.astype(np.float32)


def export_wav(audio: np.ndarray, path: Path, sample_rate: int = CONFIG.sample_rate) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)
