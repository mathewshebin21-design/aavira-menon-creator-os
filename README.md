# Aavira Menon Creator OS

Source-controlled workspace for Aavira's consistent-character image and video production.

- **GitHub Codespaces:** editing, tests and source control.
- **Kaggle / Colab:** on-demand notebook inference, subject to free runtime availability.
- **LivePortrait:** portrait and head motion from authorized driving video. Does not animate the full body or synthesize mouth shapes from standalone audio.
- **Wan / LTX:** separate image-to-video workflows. Neither runtime is tracked in this repository.

Start with [LivePortrait setup, Kaggle diagnostics and quality review](docs/LIVEPORTRAIT.md). The same notebook can run on Kaggle or Colab once the updated source is accessible in the notebook runtime.

List routes with `python -m creator_os.routes`. Run local checks with `python -m unittest discover -s tests -v` (FFmpeg required for media tests).

Keep reference images, recordings, model weights and outputs private and out of version control.
