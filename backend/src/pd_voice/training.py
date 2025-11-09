from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, roc_curve
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
    rng = np.random.default_rng(None if config.stochastic_augmentations else config.random_seed)
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
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
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
    return features, labels, metadata, extractor.get_feature_labels()


def _compute_scale_pos_weight(labels: np.ndarray) -> float | None:
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    if pos == 0 or neg == 0:
        return None
    return neg / pos


def _init_classifier(
    config: TrainingConfig,
    seed: int,
    scale_pos_weight: float | None = None,
) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        subsample=config.xgb_subsample,
        colsample_bytree=config.xgb_colsample_bytree,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=config.xgb_min_child_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig = CONFIG,
    scale_pos_weight: float | None = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    skf = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_seed,
    )
    rows = []
    diagnostics: List[Dict[str, Any]] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        clf = _init_classifier(
            config,
            seed=config.random_seed + fold,
            scale_pos_weight=scale_pos_weight,
        )
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
        fpr, tpr, _ = roc_curve(y[val_idx], proba)
        precision, recall, _ = precision_recall_curve(y[val_idx], proba)
        diagnostics.append(
            {
                "fold": fold,
                "roc": _compress_curve(fpr, tpr, "fpr", "tpr"),
                "pr": _compress_curve(recall, precision, "recall", "precision"),
            }
        )
    return pd.DataFrame(rows), diagnostics


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig = CONFIG,
    scale_pos_weight: float | None = None,
) -> Tuple[
    CalibratedClassifierCV,
    StandardScaler,
    Dict[str, float],
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
    List[float],
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
    base_clf = _init_classifier(
        config,
        seed=config.random_seed,
        scale_pos_weight=scale_pos_weight,
    )
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
    holdout_details = _build_holdout_details(y_holdout, holdout_proba)
    importance_model = _init_classifier(
        config,
        seed=config.random_seed + 999,
        scale_pos_weight=scale_pos_weight,
    )
    importance_model.fit(X_train_scaled, y_train)
    feature_importances = importance_model.feature_importances_.tolist()
    return (
        calibrated,
        scaler,
        metrics,
        train_idx,
        holdout_idx,
        holdout_details,
        feature_importances,
    )


def save_artifacts(
    model: CalibratedClassifierCV,
    scaler: StandardScaler,
    metadata: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    holdout_metrics: Dict[str, float],
    dashboard_payload: Dict[str, Any],
    config: TrainingConfig = CONFIG,
) -> Dict[str, Path]:
    artifacts = {}
    model_path = config.artifacts_dir / "pd_voice_fusion_calibrated.pkl"
    scaler_path = config.artifacts_dir / "scaler.pkl"
    metadata_path = config.artifacts_dir / "metadata.csv"
    cv_path = config.artifacts_dir / "cv_metrics.csv"
    metrics_path = config.artifacts_dir / "holdout_metrics.json"
    dashboard_path = config.artifacts_dir / "dashboard_insights.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    metadata.to_csv(metadata_path, index=False)
    cv_metrics.to_csv(cv_path, index=False)
    pd.Series(holdout_metrics).to_json(metrics_path, indent=2)
    dashboard_path.write_text(json.dumps(dashboard_payload, indent=2))

    artifacts["model"] = model_path
    artifacts["scaler"] = scaler_path
    artifacts["metadata"] = metadata_path
    artifacts["cv_metrics"] = cv_path
    artifacts["holdout_metrics"] = metrics_path
    artifacts["dashboard_insights"] = dashboard_path
    return artifacts


def _compress_curve(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_key: str,
    y_key: str,
    max_points: int = 150,
) -> List[Dict[str, float]]:
    if len(x_vals) == 0:
        return []
    step = max(1, len(x_vals) // max_points)
    return [
        {x_key: float(x_vals[i]), y_key: float(y_vals[i])}
        for i in range(0, len(x_vals), step)
    ]


def _build_holdout_details(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=10, strategy="uniform")
    hist_counts, hist_bins = np.histogram(probabilities, bins=20, range=(0, 1))
    calibration_points = [
        {"predicted": float(pred), "actual": float(true)}
        for pred, true in zip(prob_pred, prob_true)
    ]
    histogram = [
        {
            "bin": f"{hist_bins[i]:.2f}-{hist_bins[i + 1]:.2f}",
            "count": int(hist_counts[i]),
        }
        for i in range(len(hist_counts))
    ]
    return {
        "roc": _compress_curve(fpr, tpr, "fpr", "tpr"),
        "pr": _compress_curve(recall, precision, "recall", "precision"),
        "calibration": calibration_points,
        "histogram": histogram,
    }


def _aggregate_demographics(metadata: pd.DataFrame) -> Dict[str, Any]:
    base = metadata[metadata["is_augmented"] == False].copy()
    label_map = {0: "hc", 1: "pd"}
    age_bins = []
    if not base["age"].dropna().empty:
        bin_edges = [0, 40, 50, 60, 70, 80, 120]
        bin_labels = ["<40", "40-49", "50-59", "60-69", "70-79", "80+"]
        base["age_bin"] = pd.cut(base["age"], bins=bin_edges, labels=bin_labels, right=False)
        grouped = base.groupby(["age_bin", "label"]).size().unstack(fill_value=0)
        for label in bin_labels:
            counts = grouped.loc[label] if label in grouped.index else {}
            age_bins.append(
                {
                    "range": label,
                    "hc": int(counts.get(0, 0)),
                    "pd": int(counts.get(1, 0)),
                }
            )
    sex_counts = []
    base["sex"] = base["sex"].fillna("Unknown")
    grouped_sex = base.groupby(["sex", "label"]).size().unstack(fill_value=0)
    for sex, counts in grouped_sex.iterrows():
        sex_counts.append(
            {
                "sex": str(sex),
                "hc": int(counts.get(0, 0)),
                "pd": int(counts.get(1, 0)),
            }
        )
    return {
        "age_bins": age_bins,
        "sex_counts": sex_counts,
    }


def _build_embedding_points(
    features: np.ndarray,
    metadata: pd.DataFrame,
    config: TrainingConfig,
    max_samples: int = 800,
) -> List[Dict[str, Any]]:
    if len(features) == 0:
        return []
    rng = np.random.default_rng(config.random_seed)
    indices = np.arange(len(features))
    if len(indices) > max_samples:
        indices = np.sort(rng.choice(indices, size=max_samples, replace=False))
    projector = PCA(n_components=2, random_state=config.random_seed)
    coords = projector.fit_transform(features[indices])
    subset = metadata.iloc[indices].reset_index(drop=True)
    label_names = {0: "Healthy Control", 1: "Parkinson's"}
    points: List[Dict[str, Any]] = []
    for (x_val, y_val), (_, row) in zip(coords, subset.iterrows()):
        points.append(
            {
                "x": float(x_val),
                "y": float(y_val),
                "label": label_names.get(int(row["label"]), "Unknown"),
                "split": row.get("split", "train"),
            }
        )
    return points


def _prepare_feature_importance(
    feature_names: List[str],
    feature_importances: List[float],
    top_k: int = 15,
) -> List[Dict[str, Any]]:
    paired = [
        {"feature": feature_names[idx], "importance": float(value)}
        for idx, value in enumerate(feature_importances[: len(feature_names)])
    ]
    paired.sort(key=lambda item: item["importance"], reverse=True)
    return paired[:top_k]


def _build_dashboard_payload(
    cv_metrics: pd.DataFrame,
    cv_curves: List[Dict[str, Any]],
    summary_metrics: Dict[str, float],
    holdout_details: Dict[str, Any],
    metadata: pd.DataFrame,
    features: np.ndarray,
    feature_names: List[str],
    feature_importances: List[float],
    config: TrainingConfig,
) -> Dict[str, Any]:
    cv_records = cv_metrics.copy()
    cv_records["fold"] = cv_records["fold"].apply(lambda f: f"Fold {int(f)}")
    roc_payload = {
        "folds": [
            {"label": f"Fold {diag['fold']}", "points": diag["roc"]}
            for diag in cv_curves
        ],
        "holdout": {"label": "Holdout", "points": holdout_details["roc"]},
    }
    pr_payload = {
        "folds": [
            {"label": f"Fold {diag['fold']}", "points": diag["pr"]}
            for diag in cv_curves
        ],
        "holdout": {"label": "Holdout", "points": holdout_details["pr"]},
    }
    demographics = _aggregate_demographics(metadata)
    embedding_points = _build_embedding_points(features, metadata, config=config)
    feature_payload = _prepare_feature_importance(feature_names, feature_importances)
    return {
        "holdout_metrics": summary_metrics,
        "cv_metrics": cv_records.to_dict(orient="records"),
        "roc_curves": roc_payload,
        "pr_curves": pr_payload,
        "calibration": holdout_details["calibration"],
        "probability_histogram": holdout_details["histogram"],
        "feature_importance": feature_payload,
        "embedding": embedding_points,
        "demographics": demographics,
    }


def run_training_pipeline(config: TrainingConfig = CONFIG) -> Dict[str, Path]:
    samples = collect_audio_samples(config)
    prepared_examples = expand_with_augmentations(samples, config=config)
    extractor = SSLFeatureExtractor(config=config)
    features, labels, metadata, feature_names = build_feature_matrix(prepared_examples, extractor)
    scale_pos_weight = _compute_scale_pos_weight(labels)
    cv_metrics, cv_curves = cross_validate(
        features,
        labels,
        config=config,
        scale_pos_weight=scale_pos_weight,
    )
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
        holdout_details,
        feature_importances,
    ) = train_final_model(
        features,
        labels,
        config=config,
        scale_pos_weight=scale_pos_weight,
    )
    metadata = metadata.copy()
    metadata["split"] = "train"
    metadata.loc[metadata.index.isin(holdout_idx), "split"] = "holdout"
    dashboard_payload = _build_dashboard_payload(
        cv_metrics=cv_metrics,
        cv_curves=cv_curves,
        summary_metrics={**holdout_metrics, **summary},
        holdout_details=holdout_details,
        metadata=metadata,
        features=features,
        feature_names=feature_names,
        feature_importances=feature_importances,
        config=config,
    )
    artifacts = save_artifacts(
        calibrated_model,
        scaler,
        metadata,
        cv_metrics,
        {**holdout_metrics, **summary},
        dashboard_payload,
        config=config,
    )
    return artifacts
