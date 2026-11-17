"""Isolated, opt-in human portrait research workflow."""
import argparse
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from .validation import probe, stream, inspect_clip

UPSTREAM = "https://github.com/KwaiVGI/LivePortrait.git"
REV = "9b294b3d0536135442ea73cb01e6cb3ca7029dd3"
HUB = "KlingTeam/LivePortrait"
WEIGHTS_REV = "82a4fa6735ca58432b6ce39301b4b9ee066dea47"
WEIGHTS = (
    "insightface/models/buffalo_l/2d106det.onnx",
    "insightface/models/buffalo_l/det_10g.onnx",
    *(f"liveportrait/base_models/{name}.pth" for name in
      ("appearance_feature_extractor", "motion_extractor", "spade_generator", "warping_module")),
    "liveportrait/landmark.onnx",
    "liveportrait/retargeting_models/stitching_retargeting_module.pth",
)


def run(*args, **kwargs):
    print("+", shlex.join(map(str, args)), flush=True)
    return subprocess.run(args, check=True, **kwargs)


def runtime(device):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("This profile targets Linux x86_64 (Kaggle/Colab).")
    for name in ("git", "ffmpeg", "ffprobe", "uv"):
        if not shutil.which(name):
            raise ValueError(f"{name} is missing. See docs/LIVEPORTRAIT.md.")
    if device == "cuda":
        run("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv")
    print(f"Host Python {platform.python_version()}; private Python 3.11 environment for LivePortrait.")


def paths(work, device):
    return work / "LivePortrait", work / f"venv-{device}" / "bin" / "python"


def missing_weights(repo):
    missing = []
    for name in WEIGHTS:
        item = repo / "pretrained_weights" / name
        if not item.is_file() or item.stat().st_size < 1024:
            missing.append(name)
    return missing


def setup(work, device):
    runtime(device)
    work.mkdir(parents=True, exist_ok=True)
    repo, py = paths(work, device)
    if not repo.exists():
        run("git", "init", str(repo))
        run("git", "remote", "add", "origin", UPSTREAM, cwd=repo)
    if subprocess.run(["git", "cat-file", "-e", REV + "^{commit}"], cwd=repo,
                      capture_output=True).returncode:
        run("git", "fetch", "--depth", "1", UPSTREAM, REV, cwd=repo)
    run("git", "checkout", "--detach", REV, cwd=repo)
    if not py.exists():
        run("uv", "venv", "--python", "3.11", str(py.parent.parent))
    install = ("uv", "pip", "install", "--python", str(py))
    wheels = "cu121" if device == "cuda" else "cpu"
    run(*install, "--index", f"https://download.pytorch.org/whl/{wheels}",
        "torch==2.5.1", "torchvision==0.20.1")
    # The CLI doesn't import Gradio. Headless OpenCV avoids a GUI dependency and
    # shares the wheel with Albumentations. Keep upstream's remaining pins.
    raw = (repo / "requirements_base.txt").read_text().splitlines()
    req = [line.strip() for line in raw if line.strip() and not
           line.startswith(("opencv-python", "gradio"))]
    ort = "onnxruntime-gpu" if device == "cuda" else "onnxruntime"
    run(*install, *req, "opencv-python-headless==4.10.0.84",
        f"{ort}==1.19.2", "huggingface_hub==0.34.4")
    run("uv", "pip", "check", "--python", str(py))
    run(str(py), "-c",
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download({HUB!r}, revision={WEIGHTS_REV!r}, "
        f"allow_patterns={list(WEIGHTS)!r}, local_dir='pretrained_weights', max_workers=2)",
        cwd=repo)
    check(work, device)


def check(work, device):
    repo, py = paths(work, device)
    if not py.is_file():
        raise ValueError("Environment missing; run setup.")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != REV:
        raise ValueError("Unexpected LivePortrait revision; use a clean work directory.")
    missing = missing_weights(repo)
    if missing:
        raise ValueError(f"Missing or incomplete pretrained weights: {', '.join(missing)}")
    env = os.environ.copy()
    # ORT cu12/cuDNN9 uses the libraries distributed with the torch wheels.
    if device == "cuda":
        libs = sorted(py.parent.parent.glob("lib/python*/site-packages/nvidia/*/lib"))
        env["LD_LIBRARY_PATH"] = ":".join([*map(str, libs), env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    snippet = """
import sys, torch, torchvision, cv2, onnxruntime as ort
print(torch.__version__, torch.version.cuda, ort.__version__, ort.get_available_providers())
if sys.argv[1] == 'cuda':
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch cannot use the selected accelerator.')
    x = torch.ones((32, 32), device='cuda', dtype=torch.float16)
    assert (x @ x).mean().item() == 32
    provider = 'CUDAExecutionProvider'
else:
    provider = 'CPUExecutionProvider'
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
for weight in sys.argv[2:]:
    session = ort.InferenceSession(weight, sess_options=opts, providers=[provider])
    if session.get_providers()[0] != provider:
        raise RuntimeError(f'{weight}: expected {provider}; loaded {session.get_providers()}')
print('Inference providers and ONNX files verified.')
"""
    run(str(py), "-c", snippet, device,
        *(f"pretrained_weights/{name}" for name in WEIGHTS if name.endswith(".onnx")),
        cwd=repo, env=env)
    run(str(py), "inference.py", "--help", cwd=repo, env=env, stdout=subprocess.DEVNULL)
    return env


def animate(work, device, source, driving, output, fp32=False):
    source, driving, output = (p.expanduser().resolve() for p in (source, driving, output))
    if source.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not source.is_file():
        raise ValueError("Select an existing approved JPG, JPEG or PNG portrait.")
    if driving.suffix.lower() not in {".mp4", ".mov"} or not driving.is_file():
        raise ValueError("Select an existing authorized MP4 or MOV driving clip.")
    driver = probe(driving)
    if not stream(driver, "video") or not 0 < float(driver["format"]["duration"]) <= 8:
        raise ValueError("Pilot driving footage must be a valid clip of at most eight seconds.")
    env = check(work, device)
    output.mkdir(parents=True, exist_ok=False)
    repo, py = paths(work, device)
    flags = ["--flag-force-cpu"] if device == "cpu" else []
    if fp32 or device == "cpu":
        flags.append("--no-flag-use-half-precision")
    # Upstream writes a motion .pkl beside the driver. Kaggle inputs are
    # read-only, and original media directories should remain untouched.
    with tempfile.TemporaryDirectory(prefix="driver-", dir=output) as tmp:
        staged = Path(tmp) / driving.name
        staged.symlink_to(driving)
        run(str(py), "inference.py", "-s", str(source), "-d", str(staged),
            "-o", str(output), "--flag-crop-driving-video", *flags, cwd=repo, env=env)
    result = output / f"{source.stem}--{driving.stem}.mp4"
    inspect_clip(result, driving, output / "quality.json")
    print(f"Render: {result}; review: {output / 'quality.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("runtime", "setup", "check"):
        sub.add_parser(name)
    render = sub.add_parser("animate")
    for name in ("source", "driving", "output"):
        render.add_argument(f"--{name}", type=Path, required=True)
    render.add_argument("--fp32", action="store_true")
    arg = parser.parse_args()
    work = arg.work.expanduser().resolve()
    try:
        if arg.stage == "runtime":
            runtime(arg.device)
        elif arg.stage == "setup":
            setup(work, arg.device)
        elif arg.stage == "check":
            check(work, arg.device)
        else:
            animate(work, arg.device, arg.source, arg.driving, arg.output, arg.fp32)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{arg.stage} failed: {exc}\n")


if __name__ == "__main__":
    main()
