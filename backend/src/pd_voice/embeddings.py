from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from .config import CONFIG, TrainingConfig


@dataclass
class SSLModel:
    name: str
    processor: AutoFeatureExtractor
    model: AutoModel
    dim: int


class SSLFeatureExtractor:
    """Encapsulates dual SSL encoder feature extraction."""

    def __init__(
        self,
        config: TrainingConfig = CONFIG,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models: List[SSLModel] = []
        self._feature_labels: List[str] = []
        for model_name in config.ssl_models:
            processor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir=config.cache_dir)
            model = AutoModel.from_pretrained(model_name, cache_dir=config.cache_dir)
            model.to(self.device)
            model.eval()
            hidden_dim = model.config.hidden_size
            self.models.append(
                SSLModel(
                    name=model_name,
                    processor=processor,
                    model=model,
                    dim=hidden_dim,
                )
            )
            prefix = Path(model_name).name.replace("-", "_")
            for idx in range(hidden_dim):
                self._feature_labels.append(f"{prefix}_{idx+1:03d}")

    def _forward_model(self, ssl_model: SSLModel, batch_waveforms: List[np.ndarray]) -> np.ndarray:
        inputs = ssl_model.processor(
            batch_waveforms,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            outputs = ssl_model.model(input_values=input_values, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1)
            if hasattr(ssl_model.processor, "_get_feat_extract_output_lengths"):
                feat_lengths = ssl_model.processor._get_feat_extract_output_lengths(lengths)
            else:
                feat_lengths = lengths
            if not torch.is_tensor(feat_lengths):
                feat_lengths = torch.tensor(feat_lengths, device=hidden.device, dtype=hidden.dtype)
            else:
                feat_lengths = feat_lengths.to(hidden.device)
            max_len = hidden.size(1)
            frame_positions = torch.arange(max_len, device=hidden.device).unsqueeze(0)
            mask = frame_positions < feat_lengths.unsqueeze(1)
            masked_hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
            feat_lengths = feat_lengths.clamp_min(1).unsqueeze(-1)
            pooled = masked_hidden.sum(dim=1) / feat_lengths
        else:
            pooled = hidden.mean(dim=1)
        return pooled.cpu().numpy()

    def transform(
        self,
        waveforms: Iterable[np.ndarray],
        batch_size: int = 4,
    ) -> np.ndarray:
        batch_waveforms: List[np.ndarray] = []
        features: List[np.ndarray] = []
        for waveform in waveforms:
            batch_waveforms.append(waveform)
            if len(batch_waveforms) == batch_size:
                features.append(self._forward(batch_waveforms))
                batch_waveforms = []
        if batch_waveforms:
            features.append(self._forward(batch_waveforms))
        return np.vstack(features)

    def _forward(self, batch_waveforms: List[np.ndarray]) -> np.ndarray:
        per_model_features = []
        for ssl_model in self.models:
            per_model_features.append(self._forward_model(ssl_model, batch_waveforms))
        return np.concatenate(per_model_features, axis=1)

    def get_feature_labels(self) -> List[str]:
        return list(self._feature_labels)
