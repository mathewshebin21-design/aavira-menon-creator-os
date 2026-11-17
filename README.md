# Aavira Menon Creator OS

Source-controlled workspace for Aavira's consistent-character image and video production.

## Architecture
- **GitHub Codespaces:** editing, tests, source control (CPU only).
- **Kaggle:** on-demand GPU inference; Kaggle sessions and quota are limited.
- **LivePortrait:** face/head animation driven by a short, authorized driving video. Does not animate full-body movement.
- **Wan / LTX:** future image-to-video experiments; not yet validated in this repository.

## First experiment
1. Open `notebooks/Aavira_LivePortrait_Kaggle.ipynb` in Kaggle.
2. Enable Internet and GPU in Kaggle notebook settings.
3. Upload an approved Aavira source portrait and a short driving MP4 as **private** Kaggle inputs.
4. Run cells in order; inspect the output MP4 for facial drift, head jitter and blur.

**Privacy:** This repository initially contains only code and instructions, not Aavira reference images, personal recordings, or model weights. Make it private before uploading sensitive assets.

**Validation:** This is a scaffold. A GPU run and model compatibility are not yet verified.
