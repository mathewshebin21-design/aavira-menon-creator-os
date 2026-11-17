import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from creator_os.liveportrait import animate, missing_weights, WEIGHTS
from creator_os.validation import inspect_clip, review_status, RUBRIC


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class MediaTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def clip(self, name, seconds=1, audio=False):
        dest = self.root / name
        args = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x64:r=25"]
        if audio:
            args += ["-f", "lavfi", "-i", "sine=frequency=440"]
        subprocess.run([*args, "-t", str(seconds), "-c:v", "mpeg4", str(dest)], check=True)
        return dest

    def test_decodable_output_requires_quality_review(self):
        clip = self.clip("result.mp4", audio=True)
        report = self.root / "quality.json"
        data = inspect_clip(clip, clip, report)
        self.assertEqual(data["media_status"], "pass")
        self.assertEqual(review_status(data), "pending")
        for key in RUBRIC:
            data["review"][key].update(status="pass", notes="Fixture review.")
        self.assertEqual(review_status(data), "pass")
        data["review"]["lip_sync"]["status"] = "not_applicable"
        self.assertEqual(review_status(data), "pending")

    def test_rejects_missing_audio_and_trimmed_video(self):
        driver = self.clip("driver.mp4", audio=True)
        output = self.clip("result.mp4", seconds=0.5)
        report = self.root / "quality.json"
        with self.assertRaisesRegex(ValueError, "Driver audio"):
            inspect_clip(output, driver, report)
        data = json.loads(report.read_text())
        self.assertEqual(review_status(data), "fail")
        self.assertGreater(data["driver_video_duration_delta_seconds"], 0.12)

    def test_pilot_rejects_long_driver_before_environment_check(self):
        source = self.root / "portrait.png"
        source.touch()
        driver = self.clip("long.mp4", seconds=9)
        with self.assertRaisesRegex(ValueError, "eight seconds"):
            animate(self.root, "cpu", source, driver, self.root / "render")


class WeightsTest(unittest.TestCase):
    def test_only_expected_non_placeholder_weights_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in WEIGHTS:
                path = root / "pretrained_weights" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"0" * 2048)
            self.assertEqual(missing_weights(root), [])
            path.write_text("version https://git-lfs.github.com/spec/v1")
            self.assertEqual(missing_weights(root), [WEIGHTS[-1]])
