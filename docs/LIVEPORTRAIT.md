# LivePortrait research workflow

List Creator OS routes with `python -m creator_os.routes`. LivePortrait is a separate, opt-in portrait workflow; external Wan and LTX flows keep their own environments and entry points. The shared validation script can inspect output from any route.

## Kaggle session triage

The original and fresh notebook were previously reported to show “Session Error” with “Start session” disabled. The original URL returned “We can't find that page” in the current browser; the anonymous notebook API returned 403. Neither result proves exhausted GPU quota, a platform outage, or an account restriction. The earlier successful T4 probe does not establish current availability. The current saved Kaggle notebook contents could not be read.

1. In Kaggle while signed in, check **Settings → Accelerator**, the remaining GPU quota, open sessions, and any account verification notice. Record exact messages, accelerator choices, timestamp and whether an older session is still active. Check Kaggle's [official GPU usage guide](https://www.kaggle.com/docs/efficient-gpu-usage).
2. In the already-created test notebook, with no model code executing, try **None** (CPU); if it connects, evaluate `print("CPU session started")`. Repeat with GPU T4 and `!nvidia-smi` if the accelerator is offered. A CPU failure points to session provisioning/account/browser or platform issues before inference; GPU-only failure narrows to accelerator eligibility, quota, availability or GPU provisioning.
3. If GPU works and model setup fails, keep the first failing cell and its error: host tools, environment resolution, weight download, provider load, and inference are separate stages.

## Run the pilot

Use a private Kaggle or [free Colab](https://research.google.com/colaboratory/faq.html) notebook. Free Colab GPU availability and runtime limits vary. CPU is an experimental fallback for setup and short smoke tests; Codespaces is for editing/tests. A [ZeroGPU Space](https://huggingface.co/docs/hub/spaces-zerogpu) may be available subject to eligibility and quota, but the current CLI is a notebook workflow and would need a compatible Space adaptation. Keep private assets out of public demos.

Linux x86-64 needs git, FFmpeg/ffprobe and uv. From the Creator OS checkout:

```sh
python -m creator_os.liveportrait --work /tmp/lp --device cuda runtime
python -m creator_os.liveportrait --work /tmp/lp --device cuda setup
python -m creator_os.liveportrait --work /tmp/lp --device cuda animate \
  --source /tmp/lp/LivePortrait/assets/examples/source/s0.jpg \
  --driving /tmp/lp/LivePortrait/assets/examples/driving/d0.mp4 \
  --output outputs/lp-smoke-01
```

Use `--device cpu` consistently for CPU, or `--fp32` after `animate` if GPU half precision visibly corrupts the image. The bootstrap uses isolated Python 3.11, [pinned LivePortrait code](https://github.com/KlingTeam/LivePortrait/tree/9b294b3d0536135442ea73cb01e6cb3ca7029dd3) and [eight human checkpoints](https://huggingface.co/KlingTeam/LivePortrait/tree/82a4fa6735ca58432b6ce39301b4b9ee066dea47). The CUDA profile checks real FP16 computation and that the ONNX sessions select CUDA. It uses one GPU and does not assume two T4s combine memory. Stock InsightFace weights are restricted to non-commercial research; resolve model rights or replace the detector before commercial use.

## Character acceptance

Use the approved primary portrait and an authorized driver with a visible blink, one subtle nod and a spoken phrase containing visible lip closures. Keep this driver fixed when comparing settings/backends. Review the standalone result and the concat video side by side. Review all six criteria in `quality.json`, recording evidence in **notes** and timestamps where there is an issue. The mechanical check verifies decoding, video duration against the driver and whether supplied audio is present; it does not score sync or identity. A separate voice track requires a matching performance or a separate lip-sync pass.

For other backends:
```sh
python -m creator_os.validation inspect clip.mp4 --report outputs/my-review.json
python -m creator_os.validation review outputs/my-review.json
```
The review command exits nonzero until all applicable checks pass with notes. Do not commit media, reviews containing private details, or weights.
