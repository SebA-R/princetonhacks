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
```text
parkinson_voice_predictor/
+-- backend/                   # Python training + inference service
|   +-- api_server.py          # FastAPI entry point (uvicorn runner)
|   +-- requirements.txt       # Backend dependencies
|   +-- train.py / predict.py  # CLI utilities
|   +-- data/, artifacts/      # Raw + processed assets (gitignored)
|   +-- src/pd_voice/          # Core library (processing, embeddings, training, API)
+-- frontend/                  # Next.js + ShadCN dashboard
    +-- app/, components/, lib/
    +-- package.json, tsconfig.json, etc.
    +-- public/, styles/, types/
```

## Backend environment setup
1. **Enter the backend workspace (Python 3.12+):**
   `powershell
   cd backend
   python -m venv .venv
   .\.venv\Scripts\activate
   `
2. **Install dependencies (CPU is fine, CUDA accelerates SSL embedding extraction):**
   `powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   `

## Training pipeline
All steps (download  preprocessing  SSL embeddings  CV  calibration  artifact export) live inside `backend/train.py`.
`powershell
cd backend
python train.py \
  --augmentations-per-clip 2 \
  --min-duration 3.0 \
  --max-duration 8.0 \
  --test-size 0.2 \
  --cv-folds 5 \
  --seed 42
`

Key behaviors:
- **Dataset management**: Files are fetched directly from Figshare via the REST API, checksum-verified, and extracted into `backend/data/raw/{PD_AH,HC_AH}` with demographics preserved.
- **Preprocessing**: 16 kHz mono, silence trim (35 dB threshold), spectral noise suppression, -20 dBFS loudness targeting, bounded duration (38 s).
- **Augmentation**: Two stochastic variants per utterance (Gaussian noise, gain, pitch ±2 semitones, temporal shift, clipping distortion) to bolster robustness.
- **Embeddings**: Mean-pooled hidden states from Wav2Vec2-Base and WavLM-Base-Plus are concatenated, producing 1536-dim biomarkers per clip.
- **Modeling**: StandardScaler  XGBoost (600 trees, depth 8, lr 0.03, subsample 0.9)  CalibratedClassifierCV(sigmoid, 5 folds).
- **Evaluation**: 5-fold CV metrics (accuracy/AUROC ± SD) + stratified 20% holdout report saved to `backend/artifacts/holdout_metrics.json` & `backend/artifacts/cv_metrics.csv`.
- **Persistence**: `backend/artifacts/pd_voice_fusion_calibrated.pkl`, `backend/artifacts/scaler.pkl`, `backend/artifacts/metadata.csv` (sample lineage, augment flags, demographics, split).

> **Tip**: SSL model downloads (~1.3 GB total) the first time you run training/inference. Hugging Face caches them under `backend/.cache/pd_voice` for reuse.

## Inference
After training, run calibrated inference on any mono 16 kHz .wav of sustained vowel phonation from inside `backend/`:
`powershell
cd backend
python predict.py path\to\audio.wav
`

Or call the helper function directly from Python (with your backend virtualenv activated):
`python
from pd_voice import predict_voice
print(predict_voice("/path/to/audio.wav"))
# -> "PD probability (calibrated): 18.42% | band: borderline"
`

Probabilities map to risk bands used in the accompanying report:
- <10%  **low**
- 1030%  **borderline**
- >30%  **elevated**

## FastAPI service
Run the REST API that powers the dashboard and external integrations from the backend workspace.

```bash
cd backend
pip install -r requirements.txt   # if you haven't already
python api_server.py
```

Endpoints:
- `GET /health` - uptime probe.
- `POST /predict` - upload (`file`) and receive calibrated PD probability, risk band, spectrogram heatmap, and confidence.
- `POST /train` - reruns the full training pipeline and refreshes artifacts + insights JSON.
- `GET /insights` - returns dashboard-ready metrics (CV folds, ROC/PR curves, calibration, demographics, etc.).

## Frontend dashboard (Next.js + ShadCN)
A client experience for uploading phonations, monitoring inference confidence, exporting PDF summaries, and visualizing diagnostics.

```bash
cd frontend
npm install            # already run once, repeat after pulling
cp .env.example .env.local
# Update NEXT_PUBLIC_API_BASE_URL if the FastAPI server uses a different host/port
npm run dev
```

Visit http://localhost:3000 after both the frontend dev server and FastAPI backend are running.
The UI uses ShadCN primitives; add more components via `npx shadcn@latest add <component>`.

## Reproducibility & transparency
- Deterministic seeds (42) are applied to data splits, numpy RNG, and augmentation pipelines; set --seed to regenerate splits.
- Hydrated metadata (rtifacts/metadata.csv) documents every sample/augmentation, demographics (if available), and whether it landed in the final holdout.
- CV + holdout metrics, scaler, and model artefacts are versioned alongside their creation timestamps.
- Hugging Face cache + Figshare asset hashes guarantee consistent raw inputs.

## Safety disclaimer
This repository is **not** cleared for clinical use. It should inform research, prototyping, or academic exploration only. Any deployment in patient-facing scenarios **must** undergo rigorous validation, regulatory review, and ethical oversight.
