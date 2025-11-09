from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import json
import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import TrainingConfig
from .inference import VoicePredictor
from .training import run_training_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
INSIGHTS_PATH = ARTIFACTS_DIR / "dashboard_insights.json"
_STATE_LOCK = Lock()
REQUIRED_ARTIFACTS = [
    ARTIFACTS_DIR / "pd_voice_fusion_calibrated.pkl",
    ARTIFACTS_DIR / "scaler.pkl",
    INSIGHTS_PATH,
]


def build_config(overrides: Optional[Dict[str, Any]] = None) -> TrainingConfig:
    """Create a TrainingConfig rooted at the project with optional overrides."""
    overrides = overrides or {}
    base_kwargs: Dict[str, Any] = {
        "base_dir": PROJECT_ROOT,
    }
    base_kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return TrainingConfig(**base_kwargs)


def _load_predictor() -> VoicePredictor:
    _ensure_training_state()
    return VoicePredictor(artifacts_dir=ARTIFACTS_DIR)


@lru_cache(maxsize=1)
def get_predictor() -> VoicePredictor:
    return _load_predictor()


def refresh_predictor_cache() -> VoicePredictor:
    get_predictor.cache_clear()  # type: ignore[attr-defined]
    return get_predictor()


def _artifacts_missing() -> bool:
    return any(not path.exists() for path in REQUIRED_ARTIFACTS)


def _ensure_training_state(force: bool = False) -> None:
    if not force and not _artifacts_missing():
        return
    with _STATE_LOCK:
        if not force and not _artifacts_missing():
            return
        config = build_config()
        run_training_pipeline(config=config)
        get_predictor.cache_clear()  # type: ignore[attr-defined]


def _compute_confidence(probability: float) -> float:
    margin = abs(probability - 0.5) * 2  # 0 -> 0 (low confidence), 0.5 -> 1
    confidence = 0.55 + 0.45 * margin
    return max(0.55, min(1.0, confidence))


def _mel_spectrogram_matrix(waveform: np.ndarray, sample_rate: int) -> List[List[float]]:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
    return mel_norm.astype(np.float32).tolist()


def _load_insights() -> Dict[str, Any]:
    _ensure_training_state()
    if not INSIGHTS_PATH.exists():
        raise FileNotFoundError("dashboard insights have not been generated yet.")
    return json.loads(INSIGHTS_PATH.read_text())


class TrainRequest(BaseModel):
    augmentations_per_clip: Optional[int] = Field(None, ge=0, le=10)
    min_duration_sec: Optional[float] = Field(None, gt=0)
    max_duration_sec: Optional[float] = Field(None, gt=0)
    test_size: Optional[float] = Field(None, gt=0, lt=0.5)
    cv_folds: Optional[int] = Field(None, ge=2, le=10)
    seed: Optional[int] = Field(None, ge=0)


class TrainResponse(BaseModel):
    artifacts: Dict[str, str]
    metrics: Dict[str, float]
    insights: Dict[str, Any]


class PredictResponse(BaseModel):
    probability: float
    risk_band: str
    confidence: float
    spectrogram: List[List[float]]
    sample_rate: int
    duration_sec: float


app = FastAPI(title="PD Voice Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Warm-start artifacts/insights so the dashboard has telemetry immediately.
try:  # pragma: no cover - best-effort bootstrap
    _ensure_training_state()
except Exception:
    # Defer to explicit /train calls if automatic hydration fails.
    pass


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if not file.filename.lower().endswith((".wav", ".flac", ".mp3", ".m4a")):
        raise HTTPException(status_code=400, detail="Audio file must be .wav, .flac, .mp3, or .m4a")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)
    try:
        result = get_predictor().predict(tmp_path, include_waveform=True)
    except FileNotFoundError:
        _ensure_training_state(force=True)
        try:
            result = get_predictor().predict(tmp_path, include_waveform=True)
        except FileNotFoundError as exc:  # pragma: no cover - safeguard
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    waveform = result.pop("waveform", None)
    sample_rate = int(result.pop("sample_rate", 16_000))
    if waveform is None:
        raise HTTPException(status_code=500, detail="Waveform data unavailable for visualization.")
    spectrogram = _mel_spectrogram_matrix(np.asarray(waveform), sample_rate)
    confidence = _compute_confidence(float(result["probability"]))
    duration_sec = len(waveform) / sample_rate
    return PredictResponse(
        probability=float(result["probability"]),
        risk_band=str(result["risk_band"]),
        confidence=confidence * 100.0,
        spectrogram=spectrogram,
        sample_rate=sample_rate,
        duration_sec=float(duration_sec),
    )


@app.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest | None = None) -> TrainResponse:
    overrides = {}
    if request:
        overrides = request.dict()
    config = build_config(overrides)
    artifacts = run_training_pipeline(config=config)
    metrics_path = artifacts["holdout_metrics"]
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    refresh_predictor_cache()
    try:
        insights = _load_insights()
    except FileNotFoundError:
        insights = {}
    return TrainResponse(
        artifacts={k: str(v) for k, v in artifacts.items()},
        metrics=metrics,
        insights=insights,
    )


@app.get("/insights")
async def get_insights() -> Dict[str, Any]:
    try:
        return _load_insights()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
