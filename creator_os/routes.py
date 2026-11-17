"""List available Creator OS video routes without invoking any provider."""
import json

ROUTES = {
    "liveportrait": {"task": "portrait animation from performance footage",
                     "status": "research", "entrypoint": "python -m creator_os.liveportrait"},
    "wan": {"task": "image-to-video", "status": "external",
            "entrypoint": None, "note": "Continue using the separate Wan notebook; no tracked runner."},
    "ltx": {"task": "image-to-video", "status": "external",
            "entrypoint": None, "note": "No tracked LTX runtime; retain the existing external workflow."},
}

if __name__ == "__main__":
    print(json.dumps(ROUTES, indent=2))
