from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pd_voice.inference import VoicePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single audio clip.")
    parser.add_argument("audio_path", type=Path, help="Path to a .wav file.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="Directory containing scaler/model artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = VoicePredictor(artifacts_dir=args.artifacts_dir)
    result = predictor.predict(args.audio_path)
    print(
        f"PD probability (calibrated): {result['probability']*100:.2f}% | band: {result['risk_band']}"
    )


if __name__ == "__main__":
    main()

