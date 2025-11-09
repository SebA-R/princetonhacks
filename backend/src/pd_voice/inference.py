from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

from .audio_processing import preprocess_audio
from .config import CONFIG, TrainingConfig
from .embeddings import SSLFeatureExtractor
from .features import compute_acoustic_features


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
        self.transformer_path = self.artifacts_dir / "feature_pipeline.pkl"
        if not self.model_path.exists() or not self.transformer_path.exists():
            raise FileNotFoundError(
                "Missing model/scaler artifacts. Train the pipeline before inference."
            )
        self.model = joblib.load(self.model_path)
        self.transformer = joblib.load(self.transformer_path)
        self.extractor = SSLFeatureExtractor(config=self.config)

    def predict(self, audio_path: str | Path, include_waveform: bool = False) -> Dict[str, Any]:
        processed = preprocess_audio(str(audio_path), config=self.config)
        embedding = self.extractor.transform([processed], batch_size=1)
        acoustic_values, _ = compute_acoustic_features(processed, self.config.sample_rate)
        if acoustic_values.size:
            acoustic_values = acoustic_values.reshape(1, -1)
            features = np.hstack([embedding, acoustic_values])
        else:
            features = embedding
        scaled = self.transformer.transform(features)
        probability = float(self.model.predict_proba(scaled)[:, 1][0])
        result: Dict[str, Any] = {
            "probability": probability,
            "risk_band": risk_band(probability),
        }
        if include_waveform:
            result["waveform"] = processed
            result["sample_rate"] = self.config.sample_rate
        return result


def predict_voice(audio_path: str | Path) -> str:
    predictor = VoicePredictor()
    result = predictor.predict(audio_path)
    percent = result["probability"] * 100
    return f"PD probability (calibrated): {percent:.2f}% | band: {result['risk_band']}"
