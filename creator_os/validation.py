"""Local media checks and explicit human review for all video backends."""
import argparse
from fractions import Fraction
import json
import subprocess
from pathlib import Path

RUBRIC = {
    "identity": "Compare eyes, brows, nose, mouth, face shape and beauty mark to the approved portrait at start, middle and end.",
    "blinks": "Check natural closures and reopening against the driver when present; reject stuck or flickering eyes.",
    "head_motion": "Compare nods and turns to the driver; reject shakes, abrupt jumps or excessive movement.",
    "lip_sync": "With dialogue on, check mouth closure on M/B/P and check timing at the start and end. Silent clips may use not_applicable.",
    "sharpness": "Inspect eyes and lips at 100%; reject persistent softness or detail loss against the source.",
    "temporal_stability": "Scrub frame by frame around blinks and turns; reject facial warping, boundary shimmer and texture changes.",
}


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json",
        str(path.resolve())], text=True))


def stream(metadata, kind):
    return next((s for s in metadata["streams"] if s["codec_type"] == kind), None)


def inspect_clip(video, driving, report):
    meta = probe(video)
    visual = stream(meta, "video")
    audio = stream(meta, "audio")
    duration = float(meta["format"]["duration"])
    fps = float(Fraction(visual["avg_frame_rate"])) if visual and visual["avg_frame_rate"] != "0/0" else 0
    duration_delta = None
    problems = []
    if not visual or duration <= 0 or fps <= 0:
        problems.append("No usable video stream.")
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video.resolve()),
                        "-f", "null", "-"], check=True)
    except subprocess.CalledProcessError:
        problems.append("Video or audio decode failed.")
    if driving:
        source = probe(driving)
        driver_video = stream(source, "video")
        # Compare video lengths rather than container duration when available.
        def video_duration(container, track):
            return float((track or {}).get("duration", container["format"]["duration"]))
        duration_delta = abs(video_duration(meta, visual) - video_duration(source, driver_video))
        if duration_delta > max(0.12, 2 / fps if fps else 0.12):
            problems.append("Output video duration differs materially from the driver.")
        if stream(source, "audio") and not audio:
            problems.append("Driver audio is missing from the output.")
    data = {
        "media_status": "fail" if problems else "pass", "problems": problems,
        "duration_seconds": duration, "fps": fps, "driver_video_duration_delta_seconds": duration_delta,
        "width": visual.get("width") if visual else None, "height": visual.get("height") if visual else None,
        "has_audio": bool(audio), "review_status": "pending",
        "review": {k: {"status": "pending", "instruction": v, "notes": "", "timestamp_seconds": None}
                   for k, v in RUBRIC.items()},
    }
    report.write_text(json.dumps(data, indent=2) + "\n")
    if problems:
        raise ValueError("; ".join(problems))
    return data


def review_status(data):
    if data["media_status"] != "pass":
        return "fail"
    reviews = data.get("review", {})
    if any(reviews.get(key, {}).get("status") == "fail" for key in RUBRIC):
        return "fail"
    for key in RUBRIC:
        item = reviews.get(key, {})
        status = item.get("status")
        allowed = status == "pass" or (key == "lip_sync" and status == "not_applicable"
                                        and not data["has_audio"])
        if not allowed or not str(item.get("notes") or "").strip():
            return "pending"
    return "pass"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("video", type=Path)
    inspect.add_argument("--driving", type=Path)
    inspect.add_argument("--report", type=Path, required=True)
    review = sub.add_parser("review")
    review.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.action == "inspect":
        if args.report.exists():
            parser.error("Report exists; provide a fresh path to preserve review notes.")
        inspect_clip(args.video, args.driving, args.report)
    else:
        data = json.loads(args.report.read_text())
        status = data["review_status"] = review_status(data)
        args.report.write_text(json.dumps(data, indent=2) + "\n")
        print(status)
        raise SystemExit(0 if status == "pass" else 1)


if __name__ == "__main__":
    main()
