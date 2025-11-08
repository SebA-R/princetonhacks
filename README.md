# Parkinson Voice Detection Pipeline

## Overview
This repository implements a fully automated, research-grade Parkinson's voice detection workflow trained on the Figshare dataset [Voice Samples for Patients with Parkinson's Disease and Healthy Controls (ID: 23849127)](https://doi.org/10.6084/m9.figshare.23849127.v1). The system:

- Downloads and verifies the public PD/HC speech corpus and demographics sheet from Figshare.
- Applies clinic-inspired audio hygiene (silence trimming, noise reduction, loudness normalization, duration control) plus stochastic augmentations (noise, gain, pitch, shift, distortion).
- Extracts dual 768-dim SSL embeddings (facebook/wav2vec2-base + microsoft/wavlm-base-plus) and concatenates them into a 1536-dim biomarker representation.
- Trains an XGBoost classifier with Platt-scaling (CalibratedClassifierCV) on top of standardized embeddings, reporting 5-fold cross-validation (accuracy & AUROC) and an 80/20 stratified holdout set.
- Saves calibrated model/scaler/metadata for deployment-ready inference with risk-band explanations.

> **Clinical notice**: This project is for research and educational purposes only. It is **not** a medical device, diagnostic tool, or substitute for professional care.

## Project structure
``text
parkinson_voice_predictor/
+-- artifacts/                  # Saved scaler/model/metrics (gitignored)
+-- data/                       # Raw + processed data (gitignored)
+-- notebooks/                  # Optional research notebooks
+-- src/pd_voice/               # Core library
¦   +-- audio_processing.py     # Preprocessing & augmentations
¦   +-- config.py               # Global configuration and Figshare wiring
¦   +-- data_utils.py           # Download/extract + metadata parsing
¦   +-- embeddings.py           # Dual SSL feature extractor
¦   +-- inference.py            # Calibrated inference helper
¦   +-- training.py             # Training orchestration + persistence
¦   +-- __init__.py
+-- train.py                    # CLI entry point for training
+-- predict.py                  # CLI entry point for inference
+-- requirements.txt
+-- README.md
``

## Environment setup
1. **Create a virtual environment (Python 3.12+):**
   `powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   `
2. **Install dependencies (CPU is fine, CUDA accelerates SSL embedding extraction):**
   `powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   `

## Training pipeline
All steps (download ? preprocessing ? SSL embeddings ? CV ? calibration ? artifact export) are wrapped in 	rain.py:
`powershell
python train.py \
  --augmentations-per-clip 2 \
  --min-duration 3.0 \
  --max-duration 8.0 \
  --test-size 0.2 \
  --cv-folds 5 \
  --seed 42
`

Key behaviors:
- **Dataset management**: Files are fetched directly from Figshare via the REST API, checksum-verified, and extracted into data/raw/{PD_AH,HC_AH} with demographics preserved.
- **Preprocessing**: 16 kHz mono, silence trim (35 dB threshold), spectral noise suppression, -20 dBFS loudness targeting, bounded duration (3–8 s).
- **Augmentation**: Two stochastic variants per utterance (Gaussian noise, gain, pitch ±2 semitones, temporal shift, clipping distortion) to bolster robustness.
- **Embeddings**: Mean-pooled hidden states from Wav2Vec2-Base and WavLM-Base-Plus are concatenated, producing 1536-dim biomarkers per clip.
- **Modeling**: StandardScaler ? XGBoost (600 trees, depth 8, lr 0.03, subsample 0.9) ? CalibratedClassifierCV(sigmoid, 5 folds).
- **Evaluation**: 5-fold CV metrics (accuracy/AUROC ± SD) + stratified 20% holdout report saved to rtifacts/holdout_metrics.json & rtifacts/cv_metrics.csv.
- **Persistence**: rtifacts/pd_voice_fusion_calibrated.pkl, rtifacts/scaler.pkl, rtifacts/metadata.csv (sample lineage, augment flags, demographics, split).

> **Tip**: SSL model downloads (~1.3?GB total) the first time you run training/inference. Hugging Face caches them under .cache/pd_voice for reuse.

## Inference
After training, run calibrated inference on any mono 16 kHz .wav of sustained vowel phonation:
`powershell
python predict.py path\to\audio.wav
`

Or call the helper function directly from Python:
`python
from pd_voice import predict_voice
print(predict_voice("/path/to/audio.wav"))
# -> "PD probability (calibrated): 18.42% | band: borderline"
`

Probabilities map to risk bands used in the accompanying report:
- <10% ? **low**
- 10–30% ? **borderline**
- >30% ? **elevated**

## Reproducibility & transparency
- Deterministic seeds (42) are applied to data splits, numpy RNG, and augmentation pipelines; set --seed to regenerate splits.
- Hydrated metadata (rtifacts/metadata.csv) documents every sample/augmentation, demographics (if available), and whether it landed in the final holdout.
- CV + holdout metrics, scaler, and model artefacts are versioned alongside their creation timestamps.
- Hugging Face cache + Figshare asset hashes guarantee consistent raw inputs.

## Safety disclaimer
This repository is **not** cleared for clinical use. It should inform research, prototyping, or academic exploration only. Any deployment in patient-facing scenarios **must** undergo rigorous validation, regulatory review, and ethical oversight.
