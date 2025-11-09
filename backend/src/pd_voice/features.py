from __future__ import annotations

from typing import List, Tuple

import librosa
import numpy as np
import parselmouth


def compute_acoustic_features(
    waveform: np.ndarray,
    sample_rate: int,
) -> Tuple[np.ndarray, List[str]]:
    """Compute MFCC/spectral statistics plus Praat jitter/shimmer/HNR features."""
    feats: List[float] = []
    labels: List[str] = []

    def _add_stats(prefix: str, values: np.ndarray) -> None:
        arr = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        feats.append(float(arr.mean()))
        labels.append(f"{prefix}_mean")
        feats.append(float(arr.std()))
        labels.append(f"{prefix}_std")

    def _add_scalar(name: str, value: float) -> None:
        feats.append(float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)))
        labels.append(name)

    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
    for idx, coeff in enumerate(mfcc, start=1):
        _add_stats(f"mfcc_{idx:02d}", coeff)

    spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
    _add_stats("spectral_centroid", spectral_centroid)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate)
    _add_stats("spectral_bandwidth", spectral_bandwidth)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate, roll_percent=0.85)
    _add_stats("spectral_rolloff", spectral_rolloff)

    zero_cross = librosa.feature.zero_crossing_rate(waveform)
    _add_stats("zero_crossing_rate", zero_cross)

    rms = librosa.feature.rms(y=waveform)
    _add_stats("rms_energy", rms)

    spectral_flatness = librosa.feature.spectral_flatness(y=waveform)
    _add_stats("spectral_flatness", spectral_flatness)

    spectral_contrast = librosa.feature.spectral_contrast(y=waveform, sr=sample_rate, n_bands=6)
    for idx, band in enumerate(spectral_contrast, start=1):
        _add_stats(f"spectral_contrast_{idx}", band)

    try:
        sound = parselmouth.Sound(waveform, sampling_frequency=sample_rate)
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
        harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=75)
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)

        def _call_pp(function: str) -> float:
            return float(
                parselmouth.praat.call(point_process, function, 0, 0, 75, 500, 1.3)
            )

        def _call_shimmer(function: str) -> float:
            return float(
                parselmouth.praat.call([sound, point_process], function, 0, 0, 75, 500, 1.3, 1.6)
            )

        _add_scalar("jitter_local", _call_pp("Get jitter (local)"))
        _add_scalar("jitter_local_abs", _call_pp("Get jitter (local, absolute)"))
        _add_scalar("jitter_rap", _call_pp("Get jitter (rap)"))
        _add_scalar("jitter_ppq5", _call_pp("Get jitter (ppq5)"))
        _add_scalar("jitter_ddp", _call_pp("Get jitter (ddp)"))

        _add_scalar("shimmer_local", _call_shimmer("Get shimmer (local)"))
        _add_scalar("shimmer_local_db", _call_shimmer("Get shimmer (local_dB)"))
        _add_scalar("shimmer_apq3", _call_shimmer("Get shimmer (apq3)"))
        _add_scalar("shimmer_apq5", _call_shimmer("Get shimmer (apq5)"))
        _add_scalar("shimmer_apq11", _call_shimmer("Get shimmer (apq11)"))
        _add_scalar("shimmer_dda", _call_shimmer("Get shimmer (dda)"))

        hnr_values = harmonicity.values[harmonicity.values != -300]
        if hnr_values.size > 0:
            _add_scalar("hnr_mean", float(np.mean(hnr_values)))
            _add_scalar("hnr_std", float(np.std(hnr_values)))

        pitched = pitch.selected_array["frequency"]
        valid_pitch = pitched[pitched > 0]
        if valid_pitch.size > 0:
            _add_scalar("pitch_median", float(np.median(valid_pitch)))
            _add_scalar("pitch_mean", float(np.mean(valid_pitch)))
            _add_scalar("pitch_std", float(np.std(valid_pitch)))

        voiced_frames = np.count_nonzero(valid_pitch > 0)
        _add_scalar("voiced_fraction", float(voiced_frames / max(1, len(pitched))))
    except Exception:
        pass

    return np.array(feats, dtype=np.float32), labels
