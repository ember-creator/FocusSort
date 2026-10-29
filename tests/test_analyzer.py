"""Smoke tests for the analysis pipeline.  Run: python -m unittest discover tests"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "vendor"))

from analyzer import Analyzer, status_for  # noqa: E402


class AnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = Analyzer(BASE / ".testcache")
        cls.options = {"faces": True, "occlusion": True}

    @classmethod
    def tearDownClass(cls):
        pass

    def test_sharp_photo_beats_blurred_photo(self):
        sharp = self.analyzer.analyze(BASE / "demo" / "07_landscape_sharp.jpg", self.options)
        blur = self.analyzer.analyze(BASE / "demo" / "08_landscape_blur.jpg", self.options)
        self.assertGreater(sharp["score"], blur["score"] + 35)
        self.assertGreaterEqual(sharp["score"], 60)
        self.assertLess(blur["score"], 40)

    def test_portrait_face_and_occlusion(self):
        normal = self.analyzer.analyze(BASE / "demo" / "01_portrait_sharp.jpg", self.options)
        mask = self.analyzer.analyze(BASE / "demo" / "03_mask.jpg", self.options)
        sun = self.analyzer.analyze(BASE / "demo" / "04_sunglasses.jpg", self.options)
        self.assertEqual(normal["face_count"], 1)
        self.assertFalse(normal["occluded"])
        self.assertEqual(mask["face_count"], 1)
        self.assertTrue(mask["occluded"])
        self.assertEqual(sun["face_count"], 1)
        self.assertTrue(sun["occluded"])

    def test_blurred_face_is_reported(self):
        blur = self.analyzer.analyze(BASE / "demo" / "06_blur.jpg", self.options)
        self.assertLess(blur["score"], 45)
        self.assertLess(blur["faces"][0]["score"], 40)

    def test_status_threshold_is_dynamic(self):
        result = {"score": 62, "low_detail": False, "error": None, "faces": []}
        self.assertEqual(status_for(result, 60), "sharp")
        self.assertEqual(status_for(result, 65), "review")
        self.assertEqual(status_for(result, 80), "blur")
        self.assertEqual(status_for({"error": "x"}, 60), "error")
        self.assertEqual(status_for(None, 60), "pending")

    def test_cache_round_trip(self):
        path = BASE / "demo" / "02_portrait_sharp_b.jpg"
        # Ignore results left by a previous test run so this assertion is
        # independent of local cache state.
        self.analyzer.invalidate(path, self.options)
        first = self.analyzer.analyze(path, self.options)
        second = self.analyzer.analyze(path, self.options)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(first["score"], second["score"])


if __name__ == "__main__":
    unittest.main()
