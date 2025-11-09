from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pd_voice.config import TrainingConfig
from pd_voice.training import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Parkinson voice detection pipeline.")
    parser.add_argument("--base-dir", type=Path, default=PROJECT_ROOT, help="Project root directory.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for trained artifacts.",
    )
    parser.add_argument(
        "--augmentations-per-clip",
        type=int,
        default=2,
        help="Number of augmented variants to synthesize per clip.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="Minimum waveform duration after preprocessing (seconds).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=8.0,
        help="Maximum waveform duration after preprocessing (seconds).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout split fraction.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        base_dir=args.base_dir,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        augmentation_per_clip=args.augmentations_per_clip,
        min_duration_sec=args.min_duration,
        max_duration_sec=args.max_duration,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_seed=args.seed,
    )
    artifact_paths = run_training_pipeline(config=config)
    print("Training complete. Saved artifacts:")
    print(json.dumps({k: str(v) for k, v in artifact_paths.items()}, indent=2))


if __name__ == "__main__":
    main()

