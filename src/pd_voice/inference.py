from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np

from .audio_processing import preprocess_audio
from .config import CONFIG, TrainingConfig
from .embeddings import SSLFeatureExtractor


def risk_band(probability: float) -> str:
    if probability < 0.10:
        return "low"
    if probability < 0.30:
        return "borderline"
    return "elevated"


class VoicePredictor:
    """Wrapper around the trained classifier for inference."""

    def __init__(
        self,
        artifacts_dir: str | Path | None = None,
        config: TrainingConfig = CONFIG,
    ) -> None:
        self.config = config
        self.artifacts_dir = Path(artifacts_dir or config.artifacts_dir)
        self.model_path = self.artifacts_dir / "pd_voice_fusion_calibrated.pkl"
        self.scaler_path = self.artifacts_dir / "scaler.pkl"
        if not self.model_path.exists() or not self.scaler_path.exists():
            raise FileNotFoundError(
                "Missing model/scaler artifacts. Train the pipeline before inference."
            )
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.extractor = SSLFeatureExtractor(config=self.config)

    def predict(self, audio_path: str | Path) -> Dict[str, float | str]:
        processed = preprocess_audio(str(audio_path), config=self.config)
        embedding = self.extractor.transform([processed], batch_size=1)
        scaled = self.scaler.transform(embedding)
        probability = float(self.model.predict_proba(scaled)[:, 1][0])
        return {
            "probability": probability,
            "risk_band": risk_band(probability),
        }


def predict_voice(audio_path: str | Path) -> str:
    predictor = VoicePredictor()
    result = predictor.predict(audio_path)
    percent = result["probability"] * 100
    return f"PD probability (calibrated): {percent:.2f}% | band: {result['risk_band']}"

