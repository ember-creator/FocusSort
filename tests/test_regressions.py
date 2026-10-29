"""Regression tests for FocusSort v1.5.

Run from the FocusSort folder:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import types
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "vendor"))

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

import analyzer as analyzer_module  # noqa: E402
from analyzer import (  # noqa: E402
    ANALYSIS_VERSION,
    ERROR_CORRUPT,
    ERROR_EMPTY,
    ERROR_HEIF,
    ERROR_NOT_FOUND,
    Analyzer,
    find_similar_groups,
    is_heif_available,
    make_error_result,
    probe_image,
)
import app as app_module  # noqa: E402
from app import (  # noqa: E402
    DEFAULT_SETTINGS,
    MAX_RESTORE_FILES,
    Photo,
    collect_import_candidates,
    validate_restore_candidates,
)


class TemporaryWorkspace(unittest.TestCase):
    def setUp(self):
        self.work = BASE / f".test_tmp_{self.__class__.__name__}_{self._testMethodName}"
        if self.work.exists():
            shutil.rmtree(self.work)
        self.work.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)


class ErrorAndCacheTests(TemporaryWorkspace):
    def setUp(self):
        super().setUp()
        self.analyzer = Analyzer(self.work / "cache")
        self.options = {"faces": True, "occlusion": True}

    def test_missing_file_has_actionable_code_and_is_not_cached(self):
        missing = self.work / "gone.jpg"
        result = self.analyzer.analyze(missing, self.options)
        self.assertEqual(result["error_code"], ERROR_NOT_FOUND)
        self.assertIn("不存在", result["error"])
        self.assertTrue(result["error_hint"])
        self.assertEqual(list((self.work / "cache" / "meta").glob("*.json")), [])

    def test_empty_and_corrupt_files_are_distinct(self):
        empty = self.work / "empty.jpg"
        empty.write_bytes(b"")
        corrupt = self.work / "corrupt.jpg"
        corrupt.write_bytes(b"this is not a jpeg")
        self.assertEqual(probe_image(empty, deep=True)["code"], ERROR_EMPTY)
        self.assertEqual(probe_image(corrupt, deep=True)["code"], ERROR_CORRUPT)
        r1 = self.analyzer.analyze(empty, self.options)
        r2 = self.analyzer.analyze(corrupt, self.options)
        self.assertEqual(r1["error_code"], ERROR_EMPTY)
        self.assertEqual(r2["error_code"], ERROR_CORRUPT)
        self.assertEqual(list((self.work / "cache" / "meta").glob("*.json")), [])

    def test_missing_heic_explains_decoder(self):
        if is_heif_available():
            self.skipTest("pillow-heif is installed in this environment")
        fake = self.work / "phone.heic"
        fake.write_bytes(b"fake heic data")
        self.assertEqual(probe_image(fake, deep=True)["code"], ERROR_HEIF)
        result = self.analyzer.analyze(fake, self.options)
        self.assertEqual(result["error_code"], ERROR_HEIF)
        self.assertIn("pillow-heif", result["error_hint"])

    def test_cache_key_includes_version_options_mtime_and_size(self):
        image = self.work / "photo.jpg"
        Image.new("RGB", (80, 60), (120, 90, 70)).save(image)
        k1 = self.analyzer.cache_key(image, {"faces": True, "occlusion": True})
        k2 = self.analyzer.cache_key(image, {"faces": False, "occlusion": True})
        self.assertNotEqual(k1, k2)
        old = image.stat().st_mtime_ns
        os.utime(image, ns=(old, old + 1_000_000_000))
        k3 = self.analyzer.cache_key(image, {"faces": True, "occlusion": True})
        self.assertNotEqual(k1, k3)
        # Version is part of the key material as well.
        self.assertEqual(ANALYSIS_VERSION, 22)

    def test_cache_is_evicted_when_source_disappears(self):
        image = self.work / "photo.jpg"
        Image.new("RGB", (320, 240), (130, 100, 80)).save(image)
        result = self.analyzer.analyze(image, self.options)
        self.assertFalse(result["error"])
        self.assertEqual(self.analyzer.cache_stats()["entries"], 1)
        image.unlink()
        removed = self.analyzer.purge_cache(source_missing=True)
        self.assertGreaterEqual(removed["removed"], 1)
        self.assertEqual(self.analyzer.cache_stats()["entries"], 0)
        self.assertEqual(self.analyzer.cache_stats()["thumbs"], 0)

    def test_cache_age_eviction(self):
        image = self.work / "photo.jpg"
        Image.new("RGB", (320, 240), (130, 100, 80)).save(image)
        self.analyzer.analyze(image, self.options)
        meta = next((self.work / "cache" / "meta").glob("*.json"))
        old = time.time() - 45 * 86400
        os.utime(meta, (old, old))
        result = self.analyzer.purge_cache(older_than_days=30)
        self.assertGreaterEqual(result["removed"], 1)
        self.assertEqual(self.analyzer.cache_stats()["entries"], 0)

    def test_failed_result_has_no_thumb(self):
        result = self.analyzer.analyze(self.work / "missing.jpg", self.options)
        self.assertIsNone(result.get("phash"))
        self.assertEqual(result["faces"], [])


class DuplicateRegressionTests(TemporaryWorkspace):
    def test_same_photo_formats_group_but_blur_and_crop_do_not(self):
        src = Image.open(BASE / "demo" / "07_landscape_sharp.jpg").convert("RGB")
        src.save(self.work / "sharp.jpg", quality=92)
        src.resize((src.width // 2, src.height // 2), Image.Resampling.LANCZOS).save(
            self.work / "sharp.png"
        )
        src.resize((src.width * 2 // 3, src.height * 2 // 3), Image.Resampling.LANCZOS).save(
            self.work / "sharp.webp", quality=88
        )
        src.filter(ImageFilter.GaussianBlur(4)).save(self.work / "blurred.jpg", quality=92)
        src.crop((src.width // 8, src.height // 8, src.width * 7 // 8, src.height * 7 // 8)).save(
            self.work / "cropped.jpg", quality=92
        )
        a = Analyzer(self.work / "cache")
        records = []
        for p in sorted(self.work.glob("*")):
            if p.suffix.lower() not in analyzer_module.IMAGE_EXTS:
                continue
            result = a.analyze(p, {"faces": True, "occlusion": True})
            result["key"] = p.name
            records.append(result)
        groups = find_similar_groups(records)
        self.assertTrue(any({"sharp.jpg", "sharp.png", "sharp.webp"} <= set(g) for g in groups), groups)
        for group in groups:
            self.assertNotIn("blurred.jpg", group)
            self.assertNotIn("cropped.jpg", group)


class FocusAndMetadataTests(TemporaryWorkspace):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = Analyzer(BASE / ".testcache_focus")
        cls.options = {"faces": True, "occlusion": True}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(BASE / ".testcache_focus", ignore_errors=True)

    def test_focus_has_source_confidence_and_label(self):
        portrait = self.analyzer.analyze(BASE / "demo" / "01_portrait_sharp.jpg", self.options)
        landscape = self.analyzer.analyze(BASE / "demo" / "07_landscape_sharp.jpg", self.options)
        for result in (portrait, landscape):
            focus = result["focus"]
            self.assertGreaterEqual(focus["x"], 0.0)
            self.assertLessEqual(focus["x"], 1.0)
            self.assertGreaterEqual(focus["y"], 0.0)
            self.assertLessEqual(focus["y"], 1.0)
            self.assertGreaterEqual(focus["confidence"], 0.0)
            self.assertTrue(focus["label"])
        self.assertEqual(portrait["focus"]["source"], "face")
        self.assertIn("人脸", portrait["focus"]["label"])
        self.assertEqual(landscape["focus"]["source"], "sharpness")

    def test_occlusion_reason_is_exposed(self):
        mask = self.analyzer.analyze(BASE / "demo" / "03_mask.jpg", self.options)
        self.assertTrue(mask["occluded"])
        occ = mask["faces"][0]["occlusion"]
        self.assertTrue(occ["occluded"])
        self.assertTrue(occ.get("reason"))
        self.assertGreaterEqual(occ.get("score", 0), 0.48)

    def test_large_face_photo_triggers_tile_path_without_crash(self):
        # Large frames with only 1-2 faces exercise the four overlapping
        # BlazeFace tiles.  The merged tile boxes must stay 6-tuples; this was
        # the "IndexError: tuple index out of range" regression.
        src = Image.open(BASE / "demo" / "01_portrait_sharp.jpg").convert("RGB").resize((520, 620))
        canvas = Image.new("RGB", (1280, 1600), (40, 45, 55))
        canvas.paste(src, (380, 300))
        path = self.work / "large_single_face.jpg"
        canvas.save(path, quality=92)
        rgb, _, _ = self.analyzer._open_analysis_rgb(path)
        raw = self.analyzer._blaze_detect(rgb)
        self.assertTrue(all(len(box) == 6 for box in raw), raw)
        result = self.analyzer.analyze(path, self.options)
        self.assertFalse(result.get("error"), result.get("error"))
        self.assertGreaterEqual(result["face_count"], 1)
        self.assertEqual(result["focus"]["source"], "face")

    def test_group_photo_has_multiple_faces(self):
        a = Image.open(BASE / "demo" / "01_portrait_sharp.jpg").convert("RGB").resize((300, 350))
        b = Image.open(BASE / "demo" / "02_portrait_sharp_b.jpg").convert("RGB").resize((300, 350))
        group = Image.new("RGB", (800, 500), (90, 100, 110))
        group.paste(a, (50, 75))
        group.paste(b, (450, 75))
        path = self.work / "group.jpg"
        group.save(path, quality=92)
        result = self.analyzer.analyze(path, self.options)
        self.assertGreaterEqual(result["face_count"], 2)
        focus = result["focus"]
        self.assertEqual(focus["source"], "face")

    def test_exif_orientation_is_respected(self):
        image = Image.new("RGB", (40, 80), (110, 140, 90))
        exif = Image.Exif()
        exif[274] = 6  # rotate 90 CW on display
        path = self.work / "rotated.jpg"
        image.save(path, exif=exif)
        result = self.analyzer.analyze(path, {"faces": False, "occlusion": False})
        self.assertFalse(result["error"])
        self.assertEqual((result["width"], result["height"]), (80, 40))

    def test_cmyk_and_truncated_files(self):
        cmyk = self.work / "cmyk.jpg"
        Image.new("CMYK", (90, 70), (0, 50, 100, 0)).save(cmyk, quality=90)
        result = self.analyzer.analyze(cmyk, {"faces": False, "occlusion": False})
        self.assertFalse(result["error"], result.get("error"))
        self.assertEqual((result["width"], result["height"]), (90, 70))

        truncated = self.work / "truncated.jpg"
        Image.new("RGB", (400, 300), (120, 80, 40)).save(truncated, quality=92)
        raw = truncated.read_bytes()
        truncated.write_bytes(raw[: max(64, len(raw) // 3)])
        result = self.analyzer.analyze(truncated, {"faces": False, "occlusion": False})
        self.assertIsNotNone(result.get("error"))
        self.assertEqual(result["error_code"], ERROR_CORRUPT)


class AppHelperTests(unittest.TestCase):
    def test_overlay_defaults_are_off_and_restore_is_on(self):
        self.assertFalse(DEFAULT_SETTINGS["preview_focus"])
        self.assertFalse(DEFAULT_SETTINGS["preview_heatmap"])
        self.assertFalse(DEFAULT_SETTINGS["preview_faces"])
        self.assertTrue(DEFAULT_SETTINGS["restore_session"])

    def test_import_candidates_keep_explicit_missing_photo(self):
        base = BASE / ".test_import"
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir()
        good = base / "good.jpg"
        Image.new("RGB", (20, 20), (200, 50, 50)).save(good)
        bad = base / "notes.txt"
        bad.write_text("not an image", "utf-8")
        missing = base / "moved.jpg"
        files, ignored, bad_files = collect_import_candidates([base], recursive=True)
        self.assertIn(good, files)
        self.assertEqual(bad_files, [])
        self.assertGreaterEqual(ignored, 1)
        # Explicitly chosen unsupported and missing files become error cards.
        files2, _, bad2 = collect_import_candidates([bad, missing], recursive=True)
        self.assertEqual(files2, [])
        self.assertIn(bad, bad2)
        self.assertIn(missing, bad2)
        shutil.rmtree(base, ignore_errors=True)

    def test_restore_validation_cleans_missing_and_caps(self):
        work = BASE / ".test_restore"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir()
        valid = []
        for i in range(3):
            p = work / f"ok{i}.jpg"
            Image.new("RGB", (32, 24), (100 + i, 80, 60)).save(p)
            valid.append(p)
        items = [str(valid[0]), str(work / "gone1.jpg"), str(valid[1]), str(work / "gone2.jpg"), str(valid[2])]
        paths, invalid, limited = validate_restore_candidates(items, cap=2)
        self.assertEqual(invalid, 2)
        self.assertEqual(len(paths), 2)
        self.assertGreaterEqual(limited, 1)
        self.assertTrue(all(p.is_file() for p in paths))
        shutil.rmtree(work, ignore_errors=True)

    def test_photo_error_helpers_are_readable(self):
        p = Photo(path=BASE / "missing.jpg", key="k", name="missing.jpg", mtime=0, size=0)
        p.result = make_error_result(p.path, {"code": ERROR_NOT_FOUND, "reason": "文件不存在或已被移动"})
        p.state = "error"
        self.assertTrue(p.is_error)
        self.assertIn("不存在", p.error_reason())
        self.assertIn("重试", p.error_hint())


class SelectionDeletionTests(unittest.TestCase):
    """The recycle call itself is Windows-backed; here we verify program state."""

    def test_recycle_success_removes_all_selected_from_list(self):
        target = BASE / ".test_delete_target"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir()
        photos = []
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            path = target / name
            Image.new("RGB", (10, 10), (100, 100, 100)).save(path)
            photos.append(Photo(path=path, key=name, name=name, mtime=0, size=path.stat().st_size))
        app = app_module.FocusSortApp.__new__(app_module.FocusSortApp)
        app.photos = photos
        app.by_key = {p.key: p for p in photos}
        app.selected_keys = {"a.jpg", "b.jpg"}
        app.selected_key = "a.jpg"
        app.preview_window = None
        app.toast = types.SimpleNamespace(show=lambda *a, **k: None)
        app.root = types.SimpleNamespace(winfo_id=lambda: 0)
        app.inspector = types.SimpleNamespace(clear=lambda: None)
        app._rebuild_duplicates = lambda: None
        app._refresh_visible = lambda: None
        app._refresh_grid = lambda: None
        app._update_counts = lambda: None
        app.select = lambda *a, **k: None
        app.visible_photos = []
        original = app_module.recycle_files
        app_module.recycle_files = lambda paths, hwnd=0: (True, "")
        try:
            ok = app.delete_photos(["a.jpg", "b.jpg"], confirm=False)
        finally:
            app_module.recycle_files = original
            shutil.rmtree(target, ignore_errors=True)
        self.assertTrue(ok)
        self.assertEqual([p.name for p in app.photos], ["c.jpg"])
        self.assertNotIn("a.jpg", app.by_key)
        self.assertNotIn("b.jpg", app.by_key)
        self.assertEqual(app.selected_keys, set())

    def test_recycle_failure_leaves_list_untouched(self):
        target = BASE / ".test_delete_failure"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir()
        path = target / "a.jpg"
        Image.new("RGB", (10, 10), (100, 100, 100)).save(path)
        photo = Photo(path=path, key="a.jpg", name="a.jpg", mtime=0, size=path.stat().st_size)
        app = app_module.FocusSortApp.__new__(app_module.FocusSortApp)
        app.photos = [photo]
        app.by_key = {photo.key: photo}
        app.selected_keys = {"a.jpg"}
        app.selected_key = "a.jpg"
        app.preview_window = None
        app.toast = types.SimpleNamespace(show=lambda *a, **k: None)
        app.root = types.SimpleNamespace(winfo_id=lambda: 0)
        app.inspector = types.SimpleNamespace(clear=lambda: None)
        original_recycle = app_module.recycle_files
        original_error = app_module.messagebox.showerror
        app_module.recycle_files = lambda paths, hwnd=0: (False, "回收站不可用")
        app_module.messagebox.showerror = lambda *a, **k: None
        try:
            ok = app.delete_photos(["a.jpg"], confirm=False)
        finally:
            app_module.recycle_files = original_recycle
            app_module.messagebox.showerror = original_error
            shutil.rmtree(target, ignore_errors=True)
        self.assertFalse(ok)
        self.assertEqual([p.name for p in app.photos], ["a.jpg"])


if __name__ == "__main__":
    unittest.main()
