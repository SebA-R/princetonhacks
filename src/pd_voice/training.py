from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from .audio_processing import augment_waveform, preprocess_audio
from .config import CONFIG, TrainingConfig
from .data_utils import SampleRecord, collect_audio_samples
from .embeddings import SSLFeatureExtractor


@dataclass
class PreparedExample:
    sample_id: str
    label: int
    waveform: np.ndarray
    cohort: str
    source_path: Path
    is_augmented: bool = False
    augmentation_id: int = 0
    age: float | None = None
    sex: str | None = None


def expand_with_augmentations(
    samples: List[SampleRecord],
    config: TrainingConfig = CONFIG,
) -> List[PreparedExample]:
    rng = np.random.default_rng(config.random_seed)
    prepared: List[PreparedExample] = []
    for sample in samples:
        waveform = preprocess_audio(str(sample.filepath), config=config, rng=rng)
        prepared.append(
            PreparedExample(
                sample_id=sample.sample_id,
                label=sample.label,
                waveform=waveform,
                cohort=sample.cohort,
                source_path=sample.filepath,
                age=sample.age,
                sex=sample.sex,
            )
        )
        for aug_idx in range(config.augmentation_per_clip):
            augmented = augment_waveform(waveform, sample_rate=config.sample_rate, rng=rng)
            prepared.append(
                PreparedExample(
                    sample_id=f"{sample.sample_id}_aug{aug_idx+1}",
                    label=sample.label,
                    waveform=augmented,
                    cohort=sample.cohort,
                    source_path=sample.filepath,
                    is_augmented=True,
                    augmentation_id=aug_idx + 1,
                    age=sample.age,
                    sex=sample.sex,
                )
            )
    return prepared


def build_feature_matrix(
    prepared: List[PreparedExample],
    extractor: SSLFeatureExtractor,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    waveforms = [example.waveform for example in prepared]
    labels = np.array([example.label for example in prepared], dtype=np.int64)
    metadata_records = []
    features = extractor.transform(waveforms, batch_size=4)
    for idx, example in enumerate(prepared):
        metadata_records.append(
            {
                "row_id": idx,
                "sample_id": example.sample_id,
                "parent_sample_id": example.source_path.stem,
                "label": example.label,
                "cohort": example.cohort,
                "is_augmented": example.is_augmented,
                "augmentation_id": example.augmentation_id,
                "age": example.age,
                "sex": example.sex,
                "duration_sec": len(example.waveform) / extractor.config.sample_rate,
                "original_path": str(example.source_path),
            }
        )
    metadata = pd.DataFrame(metadata_records).reset_index(drop=True)
    return features, labels, metadata


def _init_classifier(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig = CONFIG,
) -> pd.DataFrame:
    skf = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_seed,
    )
    rows = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        clf = _init_classifier(config.random_seed + fold)
        clf.fit(X_train, y[train_idx])
        proba = clf.predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)
        rows.append(
            {
                "fold": fold,
                "accuracy": float(accuracy_score(y[val_idx], preds)),
                "auroc": float(roc_auc_score(y[val_idx], proba)),
            }
        )
    return pd.DataFrame(rows)


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig = CONFIG,
) -> Tuple[
    CalibratedClassifierCV,
    StandardScaler,
    Dict[str, float],
    np.ndarray,
    np.ndarray,
]:
    indices = np.arange(len(y))
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_seed,
    )
    X_train, X_holdout = X[train_idx], X[holdout_idx]
    y_train, y_holdout = y[train_idx], y[holdout_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_holdout_scaled = scaler.transform(X_holdout)
    base_clf = _init_classifier(config.random_seed)
    calibrated = CalibratedClassifierCV(
        estimator=base_clf,
        method="sigmoid",
        cv=config.cv_folds,
    )
    calibrated.fit(X_train_scaled, y_train)
    holdout_proba = calibrated.predict_proba(X_holdout_scaled)[:, 1]
    holdout_preds = (holdout_proba >= 0.5).astype(int)
    metrics = {
        "holdout_accuracy": float(accuracy_score(y_holdout, holdout_preds)),
        "holdout_auroc": float(roc_auc_score(y_holdout, holdout_proba)),
    }
    return calibrated, scaler, metrics, train_idx, holdout_idx


def save_artifacts(
    model: CalibratedClassifierCV,
    scaler: StandardScaler,
    metadata: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    holdout_metrics: Dict[str, float],
    config: TrainingConfig = CONFIG,
) -> Dict[str, Path]:
    artifacts = {}
    model_path = config.artifacts_dir / "pd_voice_fusion_calibrated.pkl"
    scaler_path = config.artifacts_dir / "scaler.pkl"
    metadata_path = config.artifacts_dir / "metadata.csv"
    cv_path = config.artifacts_dir / "cv_metrics.csv"
    metrics_path = config.artifacts_dir / "holdout_metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    metadata.to_csv(metadata_path, index=False)
    cv_metrics.to_csv(cv_path, index=False)
    pd.Series(holdout_metrics).to_json(metrics_path, indent=2)

    artifacts["model"] = model_path
    artifacts["scaler"] = scaler_path
    artifacts["metadata"] = metadata_path
    artifacts["cv_metrics"] = cv_path
    artifacts["holdout_metrics"] = metrics_path
    return artifacts


def run_training_pipeline(config: TrainingConfig = CONFIG) -> Dict[str, Path]:
    samples = collect_audio_samples(config)
    prepared_examples = expand_with_augmentations(samples, config=config)
    extractor = SSLFeatureExtractor(config=config)
    features, labels, metadata = build_feature_matrix(prepared_examples, extractor)
    cv_metrics = cross_validate(features, labels, config=config)
    mean_accuracy = cv_metrics["accuracy"].mean()
    std_accuracy = cv_metrics["accuracy"].std()
    mean_auroc = cv_metrics["auroc"].mean()
    std_auroc = cv_metrics["auroc"].std()
    summary = {
        "cv_accuracy_mean": float(mean_accuracy),
        "cv_accuracy_std": float(std_accuracy),
        "cv_auroc_mean": float(mean_auroc),
        "cv_auroc_std": float(std_auroc),
    }
    (
        calibrated_model,
        scaler,
        holdout_metrics,
        train_idx,
        holdout_idx,
    ) = train_final_model(features, labels, config=config)
    metadata = metadata.copy()
    metadata["split"] = "train"
    metadata.loc[metadata.index.isin(holdout_idx), "split"] = "holdout"
    artifacts = save_artifacts(
        calibrated_model,
        scaler,
        metadata,
        cv_metrics,
        {**holdout_metrics, **summary},
        config=config,
    )
    return artifacts
