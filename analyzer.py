"""FocusSort image analysis.

The blur score is a content-aware heuristic, not a lab measurement.  It combines
local Laplacian detail, global high-frequency energy, the radial power-spectrum
slope and face-region sharpness.  Face occlusion is a best-effort estimate based
on skin visibility, eye detection and local texture.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import base64
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps

ANALYSIS_VERSION = 22
ANALYSIS_MAX_EDGE = 1600
FACE_NORMAL_SIZE = 160
# A 180 MP file needs several hundred MB while being decoded.  Above this we
# give a useful "too large" error instead of letting the worker die with OOM.
MAX_IMAGE_PIXELS = 180_000_000
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
HEIF_EXTS = {".heic", ".heif"}
MODEL_DIR = Path(__file__).resolve().parent / "models"
BLAZE_MODEL = MODEL_DIR / "face_detection_short_range.tflite"

# Error codes are deliberately stable; the UI maps them to short, actionable
# messages while error_hint explains what the user can do next.
ERROR_NOT_FOUND = "file_not_found"
ERROR_PERMISSION = "permission_denied"
ERROR_UNSUPPORTED = "format_unsupported"
ERROR_HEIF = "heif_decoder_missing"
ERROR_EMPTY = "file_empty"
ERROR_CORRUPT = "file_corrupt"
ERROR_TOO_LARGE = "image_too_large"
ERROR_MEMORY = "out_of_memory"
ERROR_ANALYSIS = "analysis_failed"

ERROR_ACTIONS = {
    ERROR_NOT_FOUND: "文件可能被删除、移动或改名。请点「重试」；如果已移动，可「从列表移除」后重新导入。",
    ERROR_PERMISSION: "请检查文件/文件夹权限，或关闭占用该文件的程序后点「重试」。",
    ERROR_UNSUPPORTED: "支持 JPG / PNG / WEBP / BMP / TIFF；HEIC/HEIF 需要 pillow-heif 解码器。请转换格式后重试。",
    ERROR_HEIF: "当前没有安装 HEIC/HEIF 解码器。安装 pillow-heif，或先用系统「照片」导出为 JPG 后重新导入。",
    ERROR_EMPTY: "文件大小为 0 字节，可能没有复制完成。请重新复制原文件后点「重试」。",
    ERROR_CORRUPT: "文件头或图像数据损坏，可能复制/下载不完整。请重新获取该文件后点「重试」。",
    ERROR_TOO_LARGE: "图片像素过大，超出本机安全内存上限。请先缩小副本再分析（原图不会被修改）。",
    ERROR_MEMORY: "分析时内存不足。请关闭其他程序，或缩小图片后点「重试」。",
    ERROR_ANALYSIS: "图像分析出现内部错误。可点「重试」；若持续出现，请查看 error.log。",
}


def is_heif_available() -> bool:
    """Return True when Pillow can actually open HEIC/HEIF files."""
    return globals().get("_HEIF_AVAILABLE", False)


def _register_heif_opener() -> bool:
    try:
        from pillow_heif import register_heif_opener  # type: ignore

        register_heif_opener()
        return True
    except Exception:
        return False


_HEIF_AVAILABLE = _register_heif_opener()


class ImageTooLargeError(Exception):
    """Raised before decoding an image that would exceed the memory budget."""


def _error(code: str, reason: str, detail: str = "", hint: str | None = None) -> dict[str, Any]:
    return {
        "code": code,
        "reason": reason,
        "detail": detail,
        "hint": hint or ERROR_ACTIONS.get(code, ""),
    }


def make_error_result(path: str | Path, info: dict[str, Any], width: int = 0, height: int = 0) -> dict[str, Any]:
    """Build a normal analysis result describing an unreadable file.

    Keeping the same top-level shape as a successful result means the grid,
    filters, export and inspector do not need a separate error object.
    """
    try:
        source = str(Path(path))
    except (TypeError, ValueError):
        source = str(path)
    return {
        "version": ANALYSIS_VERSION,
        "path": source,
        "error": info.get("reason") or "无法读取图片",
        "error_code": info.get("code") or ERROR_ANALYSIS,
        "error_hint": info.get("hint") or ERROR_ACTIONS.get(ERROR_ANALYSIS, ""),
        "error_detail": info.get("detail") or "",
        "retryable": True,
        "width": int(width or 0),
        "height": int(height or 0),
        "score": 0.0,
        "global_score": 0.0,
        "face_score": None,
        "faces": [],
        "face_count": 0,
        "occluded": False,
        "occlusion_reasons": [],
        "status": "error",
        "low_detail": False,
        "focus": None,
        "metrics": {},
        "phash": None,
        "fingerprint": None,
    }


def describe_exception(exc: BaseException, path: str | Path | None = None) -> dict[str, Any]:
    """Translate a filesystem/Pillow/OpenCV exception into a user-actionable error."""
    suffix = Path(path).suffix.lower() if path is not None else ""
    detail = f"{type(exc).__name__}: {exc}"[:1200]

    if isinstance(exc, FileNotFoundError):
        return _error(ERROR_NOT_FOUND, "文件不存在或已被移动", detail)
    if isinstance(exc, PermissionError):
        return _error(ERROR_PERMISSION, "没有读取权限", detail)
    if isinstance(exc, MemoryError):
        return _error(ERROR_MEMORY, "内存不足", detail)
    if isinstance(exc, ImageTooLargeError):
        return _error(ERROR_TOO_LARGE, str(exc) or "图片像素过大", detail)
    if isinstance(exc, Image.DecompressionBombError):
        return _error(ERROR_TOO_LARGE, "图片像素过大，已停止解码", detail)

    text = str(exc).lower()
    if "truncated" in text or "broken data" in text or "corrupt" in text or "crc" in text:
        return _error(ERROR_CORRUPT, "文件损坏或图像数据不完整", detail)
    if "permission" in text or "access is denied" in text or "拒绝访问" in text:
        return _error(ERROR_PERMISSION, "没有读取权限", detail)
    if "cannot identify image file" in text or "unidentifiedimage" in type(exc).__name__.lower():
        if suffix in HEIF_EXTS and not is_heif_available():
            return _error(ERROR_HEIF, "HEIC/HEIF 解码器未安装", detail)
        if suffix in IMAGE_EXTS:
            return _error(ERROR_CORRUPT, "文件已损坏或不是有效的图片", detail)
        return _error(ERROR_UNSUPPORTED, "格式不支持", detail)
    if "out of memory" in text or "not enough memory" in text:
        return _error(ERROR_MEMORY, "内存不足", detail)
    if suffix in HEIF_EXTS and not is_heif_available():
        return _error(ERROR_HEIF, "HEIC/HEIF 解码器未安装", detail)
    return _error(ERROR_ANALYSIS, "无法读取或分析该图片", detail)


def probe_image(path: str | Path, *, deep: bool = False) -> dict[str, Any]:
    """Lightweight import/restore probe.

    ``deep=False`` is intentionally cheap: extension, existence, readability
    and size only.  It is used before queueing thousands of imports.
    ``deep=True`` additionally opens and verifies the image header; startup
    session recovery uses it so moved/corrupt files are retired immediately.
    """
    try:
        p = Path(path)
    except (TypeError, ValueError) as exc:
        return _error(ERROR_UNSUPPORTED, "路径无效", str(exc))
    suffix = p.suffix.lower()
    if suffix not in IMAGE_EXTS:
        if suffix in HEIF_EXTS and not is_heif_available():
            return _error(ERROR_HEIF, "HEIC/HEIF 解码器未安装", f"extension={suffix}")
        return _error(ERROR_UNSUPPORTED, "不支持的图片格式", f"extension={suffix or '(none)'}")
    try:
        if not p.is_file():
            return _error(ERROR_NOT_FOUND, "文件不存在或已被移动", str(p))
        st = p.stat()
    except FileNotFoundError as exc:
        return _error(ERROR_NOT_FOUND, "文件不存在或已被移动", str(exc))
    except PermissionError as exc:
        return _error(ERROR_PERMISSION, "没有读取权限", str(exc))
    except OSError as exc:
        return describe_exception(exc, p)
    if st.st_size <= 0:
        return _error(ERROR_EMPTY, "文件为空（0 字节）", str(p))
    if suffix in HEIF_EXTS and not is_heif_available():
        return _error(ERROR_HEIF, "HEIC/HEIF 解码器未安装", f"{p.name}: pillow-heif 不可用")
    if not os.access(p, os.R_OK):
        return _error(ERROR_PERMISSION, "没有读取权限", str(p))
    if deep:
        try:
            with Image.open(p) as im:
                if im.width <= 0 or im.height <= 0:
                    return _error(ERROR_CORRUPT, "图片尺寸无效", str(p))
                if im.width * im.height > MAX_IMAGE_PIXELS:
                    return _error(
                        ERROR_TOO_LARGE,
                        f"图片像素过大（{im.width * im.height / 1_000_000:.0f} MP）",
                        f"{im.width}x{im.height}",
                    )
                try:
                    im.verify()
                except (AttributeError, NotImplementedError):
                    pass
        except Exception as exc:  # noqa: BLE001 - converted to a readable error
            if suffix in HEIF_EXTS and not is_heif_available():
                return _error(ERROR_HEIF, "HEIC/HEIF 解码器未安装", str(exc))
            return describe_exception(exc, p)
    return {"code": "ok", "reason": "", "detail": "", "hint": ""}


def is_supported_image_path(path: str | Path) -> bool:
    try:
        return Path(path).suffix.lower() in IMAGE_EXTS
    except (TypeError, ValueError):
        return False


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def _log10(v: float) -> float:
    return math.log10(max(0.0, v) + 1.0)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


class Analyzer:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            (self.cache_dir / "thumbs").mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "meta").mkdir(parents=True, exist_ok=True)

        self._cascade_dir = Path(cv2.data.haarcascades)
        self._thread_local = threading.local()
        self._blaze_load_lock = threading.Lock()
        self._blaze_ok = BLAZE_MODEL.exists()
        self.last_face_backend = "blazeface+haar" if self._blaze_ok else "haar"

    def _cascades(self) -> tuple[cv2.CascadeClassifier, cv2.CascadeClassifier, cv2.CascadeClassifier, cv2.CascadeClassifier]:
        """Haar cascades are not safe to share between worker threads."""
        bundle = getattr(self._thread_local, "cascades", None)
        if bundle is None:
            bundle = (
                cv2.CascadeClassifier(str(self._cascade_dir / "haarcascade_frontalface_default.xml")),
                cv2.CascadeClassifier(str(self._cascade_dir / "haarcascade_profileface.xml")),
                cv2.CascadeClassifier(str(self._cascade_dir / "haarcascade_eye.xml")),
                cv2.CascadeClassifier(str(self._cascade_dir / "haarcascade_eye_tree_eyeglasses.xml")),
            )
            self._thread_local.cascades = bundle
        return bundle

    # ------------------------------------------------------------------ cache
    def cache_key(self, path: Path, options: dict[str, Any]) -> str:
        st = path.stat()
        # Include every option that can alter the result, not only the two
        # current switches, so future options cannot silently reuse old data.
        options_sig = json.dumps({str(k): v for k, v in sorted((options or {}).items())}, sort_keys=True, default=str)
        raw = "|".join(
            [
                str(ANALYSIS_VERSION),
                path.resolve().as_posix().lower(),
                str(st.st_mtime_ns),
                str(st.st_size),
                options_sig,
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]

    def _cache_paths(self, key: str) -> tuple[Path, Path] | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / "meta" / f"{key}.json", self.cache_dir / "thumbs" / f"{key}.jpg"

    def thumbnail_path(self, key: str) -> Path | None:
        p = self._cache_paths(key)
        if p and p[1].exists():
            return p[1]
        return None

    def _touch_cache(self, meta_path: Path) -> None:
        """Update the cache's last-used time for age-based eviction."""
        try:
            os.utime(meta_path, None)
        except OSError:
            pass

    def _read_cache_meta(self, meta_path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(meta_path.read_text("utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        return data

    def invalidate(self, path: str | Path, options: dict[str, Any] | None = None) -> int:
        """Drop cache entries for one source path (or one exact option set)."""
        if not self.cache_dir:
            return 0
        removed = 0
        if options is not None:
            try:
                key = self.cache_key(Path(path), options)
            except OSError:
                return 0
            paths = self._cache_paths(key)
            if paths:
                for p in paths:
                    try:
                        if p.exists():
                            p.unlink()
                            removed += 1
                    except OSError:
                        pass
            return removed
        try:
            target = Path(path).resolve().as_posix().lower()
        except OSError:
            target = str(path).replace("\\", "/").lower()
        for meta_path in (self.cache_dir / "meta").glob("*.json"):
            data = self._read_cache_meta(meta_path)
            if not data:
                continue
            try:
                cached = Path(str(data.get("path", ""))).resolve().as_posix().lower()
            except (OSError, ValueError):
                cached = str(data.get("path", "")).replace("\\", "/").lower()
            if cached != target:
                continue
            try:
                if meta_path.exists():
                    meta_path.unlink()
                    removed += 1
                thumb = self.cache_dir / "thumbs" / f"{meta_path.stem}.jpg"
                if thumb.exists():
                    thumb.unlink()
                    removed += 1
            except OSError:
                pass
        return removed

    def cache_stats(self) -> dict[str, Any]:
        if not self.cache_dir:
            return {"entries": 0, "thumbs": 0, "bytes": 0, "missing_sources": 0}
        meta_dir, thumb_dir = self.cache_dir / "meta", self.cache_dir / "thumbs"
        entries = 0
        missing = 0
        total = 0
        for meta_path in meta_dir.glob("*.json"):
            entries += 1
            try:
                total += meta_path.stat().st_size
            except OSError:
                pass
            data = self._read_cache_meta(meta_path)
            raw_path = (data or {}).get("path")
            if raw_path and not self._source_exists(str(raw_path)):
                missing += 1
        thumbs = 0
        for thumb in thumb_dir.glob("*.jpg"):
            thumbs += 1
            try:
                total += thumb.stat().st_size
            except OSError:
                pass
        return {"entries": entries, "thumbs": thumbs, "bytes": total, "missing_sources": missing}

    @staticmethod
    def _source_exists(raw_path: str) -> bool:
        try:
            p = Path(raw_path)
            if p.is_file():
                return True
            if p.is_absolute():
                return False
            # Old caches may contain a path relative to the app folder.
            return (Path(__file__).resolve().parent / p).is_file()
        except (OSError, ValueError):
            return False

    def purge_cache(
        self,
        *,
        remove_all: bool = False,
        source_missing: bool = False,
        older_than_days: float | None = None,
        keep_keys: Iterable[str] | None = None,
    ) -> dict[str, int]:
        """Evict cache records without ever touching the source photos.

        - ``source_missing``: remove metadata/thumbs whose original file is gone.
        - ``older_than_days``: remove records unused for that many days.
        - ``remove_all``: remove every cache record.
        Orphan thumbs are cleaned in all modes.
        """
        if not self.cache_dir:
            return {"removed": 0, "bytes": 0}
        keep = set(keep_keys or ())
        cutoff = time.time() - max(0.0, float(older_than_days)) * 86400.0 if older_than_days is not None else None
        removed = 0
        freed = 0
        meta_dir, thumb_dir = self.cache_dir / "meta", self.cache_dir / "thumbs"
        for meta_path in list(meta_dir.glob("*.json")):
            key = meta_path.stem
            if key in keep:
                continue
            data = self._read_cache_meta(meta_path)
            delete = bool(remove_all) or data is None
            if not delete and cutoff is not None:
                try:
                    delete = meta_path.stat().st_mtime < cutoff
                except OSError:
                    delete = True
            if not delete and source_missing:
                raw = (data or {}).get("path", "")
                if not self._source_exists(str(raw)):
                    delete = True
            if not delete:
                continue
            thumb = thumb_dir / f"{key}.jpg"
            for p in (meta_path, thumb):
                try:
                    if p.exists():
                        freed += p.stat().st_size
                        p.unlink()
                        removed += 1
                except OSError:
                    pass
        # Orphan thumbnails after a crash or manual deletion of metadata.
        for thumb in list(thumb_dir.glob("*.jpg")):
            if thumb.stem in keep:
                continue
            if not (meta_dir / f"{thumb.stem}.json").exists():
                try:
                    freed += thumb.stat().st_size
                    thumb.unlink()
                    removed += 1
                except OSError:
                    pass
        return {"removed": removed, "bytes": freed}

    # ------------------------------------------------------------------ public
    def analyze(self, path: str | Path, options: dict[str, Any] | None = None) -> dict[str, Any]:
        path = Path(path)
        options = options or {}
        t0 = time.perf_counter()
        probe = probe_image(path, deep=False)
        if not probe.get("code") == "ok":
            data = make_error_result(path, probe)
            data["cache_key"] = ""
            data["cached"] = False
            data["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return data

        key = ""
        try:
            key = self.cache_key(path, options)
        except OSError as exc:
            data = make_error_result(path, describe_exception(exc, path))
            data["cached"] = False
            data["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return data

        if key:
            paths = self._cache_paths(key)
            if paths and paths[0].exists():
                data = None
                try:
                    data = json.loads(paths[0].read_text("utf-8"))
                except (OSError, ValueError):
                    data = None
                if (
                    isinstance(data, dict)
                    and data.get("version") == ANALYSIS_VERSION
                    and not data.get("error")
                    and str(data.get("path", "")) != ""
                ):
                    self._touch_cache(paths[0])
                    data["cache_key"] = key
                    data["cached"] = True
                    data["elapsed_ms"] = 0
                    return data
                # Corrupt/old/error cache must not poison retries.
                for stale in paths:
                    try:
                        stale.unlink()
                    except OSError:
                        pass

        data = self._analyze_uncached(path, options)
        data["cache_key"] = key
        data["cached"] = False
        data["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if key and not data.get("error"):
            self._save_cache(key, data)
        return data

    def _save_cache(self, key: str, data: dict[str, Any]) -> None:
        paths = self._cache_paths(key)
        if not paths or data.get("error"):
            return
        meta_path, thumb_path = paths
        try:
            thumb = data.get("_thumb_rgb")
            if thumb is not None:
                Image.fromarray(thumb).save(thumb_path, "JPEG", quality=88, optimize=True)
            payload = {k: v for k, v in data.items() if not k.startswith("_")}
            payload["cached_at"] = int(time.time())
            tmp = meta_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), "utf-8")
            os.replace(tmp, meta_path)
        except OSError:
            pass

    # ------------------------------------------------------------- image basics
    @staticmethod
    def _open_analysis_rgb(path: Path) -> tuple[np.ndarray, int, int]:
        with Image.open(path) as im:
            if im.width <= 0 or im.height <= 0:
                raise OSError("图片尺寸无效")
            original_size = im.size
            orientation = 1
            try:
                orientation = int(im.getexif().get(274, 1) or 1)
            except Exception:
                pass
            if orientation in (5, 6, 7, 8):
                original_size = (original_size[1], original_size[0])
            if int(original_size[0]) * int(original_size[1]) > MAX_IMAGE_PIXELS:
                mp = int(original_size[0]) * int(original_size[1]) / 1_000_000
                raise ImageTooLargeError(f"图片像素过大（{mp:.0f} MP），超过 {MAX_IMAGE_PIXELS / 1_000_000:.0f} MP 安全上限")

            # JPEG draft decoding is much faster on large camera files.
            try:
                im.draft("RGB", (ANALYSIS_MAX_EDGE * 2, ANALYSIS_MAX_EDGE * 2))
            except Exception:
                pass
            im = ImageOps.exif_transpose(im)
            # CMYK / P / LA / I;16 and other modes are normalised before the
            # OpenCV pipeline, which only accepts 8-bit RGB/BGR.
            if im.mode != "RGB":
                im = im.convert("RGB")
            if max(im.size) > ANALYSIS_MAX_EDGE:
                im.thumbnail((ANALYSIS_MAX_EDGE, ANALYSIS_MAX_EDGE), Image.Resampling.LANCZOS)
            # .copy() detaches the array from the Pillow buffer before close().
            rgb = np.asarray(im, dtype=np.uint8).copy()
        if rgb.size == 0 or rgb.ndim != 3:
            raise OSError("图片没有可用的像素数据")
        return rgb, int(original_size[0]), int(original_size[1])

    # --------------------------------------------------------------- metrics
    @staticmethod
    def _spectral_slope(gray: np.ndarray) -> float:
        h, w = gray.shape
        s = min(h, w)
        if s < 64:
            return -3.0
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        crop = gray[y0 : y0 + s, x0 : x0 + s]
        n = 384
        crop = cv2.resize(crop, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
        crop -= float(crop.mean())
        win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
        power = np.abs(np.fft.fftshift(np.fft.fft2(crop * win))) ** 2
        yy, xx = np.mgrid[-n // 2 : n // 2, -n // 2 : n // 2]
        radius = np.sqrt(xx * xx + yy * yy).astype(np.int32)
        counts = np.bincount(radius.ravel()).astype(np.float32)
        sums = np.bincount(radius.ravel(), weights=power.ravel()).astype(np.float32)
        radial = sums / np.maximum(counts, 1)
        freqs = np.arange(radial.size, dtype=np.float32) / n
        mask = (freqs > 0.025) & (freqs < 0.44) & (radial > 1e-9)
        if int(mask.sum()) < 10:
            return -3.0
        slope = np.polyfit(np.log(freqs[mask]), np.log(radial[mask]), 1)[0]
        return float(slope)

    @staticmethod
    def _phash(gray: np.ndarray) -> str | None:
        """64-bit perceptual hash for finding the same photo in other formats."""
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        if float(small.std()) < 3.0:
            return None
        low = cv2.dct(small)[:16, :16].reshape(-1)
        median = float(np.median(low[1:]))
        value = 0
        for bit in (low > median):
            value = (value << 1) | int(bit)
        return f"{value:064x}"

    def _global_metrics(self, rgb: np.ndarray) -> dict[str, Any]:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gf = gray.astype(np.float32)

        lap = cv2.Laplacian(gf, cv2.CV_32F, ksize=3)
        lap_var = float(lap.var())

        k = 31
        mean = cv2.boxFilter(lap, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
        mean_sq = cv2.boxFilter(lap * lap, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
        var_map = np.maximum(mean_sq - mean * mean, 0.0)
        small = cv2.resize(var_map, (96, 96), interpolation=cv2.INTER_AREA)
        p90 = float(np.percentile(small, 90))
        p50 = float(np.percentile(small, 50))

        gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        edge_density = float(np.mean(mag > 25.0))

        hf = gf - cv2.GaussianBlur(gf, (0, 0), 1.2)
        hf_ratio = float(hf.std() / (gf.std() + 1e-6))
        slope = self._spectral_slope(gray)

        detail = 0.62 * _log10(p90) + 0.38 * _log10(lap_var)
        score = 100.0 * _clip((detail - 0.95) / 2.55, 0.0, 1.0)
        softness = _clip((-slope - 3.0) / 2.0, 0.0, 1.0)
        score *= 1.0 - 0.20 * softness
        # High-frequency energy is a second, contrast-robust blur cue: a soft
        # photo can still have a few strong edges but loses the fine band.
        score *= 0.85 + 0.15 * _clip(hf_ratio / 0.20)

        confidence = _clip(0.58 * _clip(p90 / 260.0) + 0.42 * _clip(edge_density / 0.055))
        low_detail = p90 < 140.0 and edge_density < 0.030

        # Estimate the most likely focus region from the strongest local-detail
        # cluster.  This is an optical inference, not the camera's AF metadata.
        peak = float(small.max())
        threshold = max(float(np.percentile(small, 92)), peak * 0.55)
        focus_conf = 0.0
        focus_xy = [0.5, 0.5]
        if peak > 5 and threshold > 0:
            ys, xs = np.nonzero(small >= threshold)
            if xs.size >= 3:
                weights = small[ys, xs].astype(np.float64)
                total = float(weights.sum())
                if total > 0:
                    focus_xy = [
                        round(float(((xs + 0.5) * weights).sum()) / total / small.shape[1], 4),
                        round(float(((ys + 0.5) * weights).sum()) / total / small.shape[0], 4),
                    ]
                    focus_conf = _clip((peak - p50) / max(peak, 1.0)) * _clip(xs.size / 40.0)
        focus = {
            "x": focus_xy[0],
            "y": focus_xy[1],
            "source": "sharpness",
            "confidence": round(float(focus_conf), 3),
            "label": "细节",
        }

        return {
            "lap_var": round(lap_var, 2),
            "detail_p90": round(p90, 2),
            "detail_p50": round(p50, 2),
            "edge_density": round(edge_density, 4),
            "hf_ratio": round(hf_ratio, 4),
            "spectral_slope": round(slope, 3),
            "phash": self._phash(gray),
            "contrast": round(float(gf.std()), 2),
            "global_score": round(_clip(score, 0, 100), 1),
            "confidence": round(confidence, 3),
            "low_detail": bool(low_detail),
            "focus": focus,
        }

    # ----------------------------------------------------------------- faces
    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        return inter / float(aw * ah + bw * bh - inter)

    def _merge_boxes(
        self, boxes: list[tuple[int, int, int, int, float, list[tuple[float, float]] | None]]
    ) -> list[tuple[tuple[int, int, int, int], list[tuple[float, float]] | None]]:
        # Be defensive: a malformed detection must never take down the
        # analysis of an otherwise valid photo.
        boxes = [b for b in boxes if len(b) >= 6]
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        kept: list[list[Any]] = []
        for x, y, w, h, weight, landmarks in boxes:
            if w < 24 or h < 24:
                continue
            if not (0.66 <= w / max(h, 1) <= 1.58):
                continue
            duplicate = False
            for item in kept:
                if self._iou((x, y, w, h), item[0]) > 0.34:
                    if item[1] is None and landmarks is not None:
                        item[1] = landmarks
                    duplicate = True
                    break
            if not duplicate:
                kept.append([(x, y, w, h), landmarks])
        return [(item[0], item[1]) for item in kept[:20]]

    def _blaze_detector(self):
        net = getattr(self._thread_local, "blaze_net", None)
        if net is False:
            return None
        if net is None:
            try:
                # OpenCV's TFLite importer is not safe to initialise from
                # several threads at once.  Serialise the one-time load.
                with self._blaze_load_lock:
                    net = cv2.dnn.readNetFromTFLite(str(BLAZE_MODEL))
                self._thread_local.blaze_net = net
            except Exception:
                self._thread_local.blaze_net = False
                self._blaze_ok = False
                return None
        return net

    @staticmethod
    def _blaze_anchors() -> np.ndarray:
        cached = getattr(Analyzer, "_anchors_cache", None)
        if cached is not None:
            return cached
        anchors: list[tuple[float, float]] = []
        # MediaPipe face_detection_short_range: 128px, strides 8/16/16/16,
        # two anchors per cell.  Total 896 anchors.
        for stride in (8, 16, 16, 16):
            fm = 128 // stride
            for y in range(fm):
                for x in range(fm):
                    for _ in range(2):
                        anchors.append(((x + 0.5) / fm, (y + 0.5) / fm))
        arr = np.asarray(anchors, dtype=np.float32)
        Analyzer._anchors_cache = arr
        return arr

    def _blaze_detect(
        self, rgb: np.ndarray, _tile: bool = False
    ) -> list[tuple[int, int, int, int, float, list[tuple[float, float]] | None]]:
        if not self._blaze_ok:
            return []
        net = self._blaze_detector()
        if net is None:
            return []
        h, w = rgb.shape[:2]
        size = 128
        scale = min(size / w, size / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(rgb, (nw, nh), interpolation=interp)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        x0, y0 = (size - nw) // 2, (size - nh) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
        tensor = canvas.astype(np.float32) / 127.5 - 1.0
        blob = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
        try:
            net.setInput(blob)
            scores, raw_boxes = net.forward(["classificators", "regressors"])
        except cv2.error:
            return []
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -80.0, 80.0)))
        raw = np.asarray(raw_boxes, dtype=np.float32).reshape(-1, 8, 2) / float(size)
        anchors = self._blaze_anchors()
        if raw.shape[0] != anchors.shape[0]:
            return []
        raw[:, 0] += anchors
        for i in range(2, 8):
            raw[:, i] += anchors
        center = raw[:, 0]
        half = raw[:, 1] / 2.0
        xmin = center[:, 0] - half[:, 0]
        ymin = center[:, 1] - half[:, 1]
        xmax = center[:, 0] + half[:, 0]
        ymax = center[:, 1] + half[:, 1]

        pad_x = (size - nw) / 2.0 / size
        pad_y = (size - nh) / 2.0 / size
        content_x = nw / size
        content_y = nh / size

        def unpad_x(v: float) -> float:
            return _clip((v - pad_x) / max(content_x, 1e-5), 0.0, 1.0)

        def unpad_y(v: float) -> float:
            return _clip((v - pad_y) / max(content_y, 1e-5), 0.0, 1.0)

        candidates: list[tuple[float, int, int, int, int, list[tuple[float, float]]]] = []
        for i in np.where(scores >= 0.72)[0]:
            if xmax[i] <= xmin[i] or ymax[i] <= ymin[i]:
                continue
            nx1, nx2 = unpad_x(float(xmin[i])), unpad_x(float(xmax[i]))
            ny1, ny2 = unpad_y(float(ymin[i])), unpad_y(float(ymax[i]))
            bw, bh = int((nx2 - nx1) * w), int((ny2 - ny1) * h)
            if bw < 24 or bh < 24:
                continue
            kps: list[tuple[float, float]] = []
            for j in range(6):
                kx = unpad_x(float(raw[i, 2 + j, 0])) * w
                ky = unpad_y(float(raw[i, 2 + j, 1])) * h
                kps.append((kx, ky))
            candidates.append((float(scores[i]), int(nx1 * w), int(ny1 * h), bw, bh, kps))
        candidates.sort(reverse=True, key=lambda c: c[0])
        kept: list[tuple[int, int, int, int, float, list[tuple[float, float]] | None]] = []
        for score, x, y, bw, bh, kps in candidates:
            box = (x, y, bw, bh)
            if any(self._iou(box, k[:4]) > 0.30 for k in kept):
                continue
            kept.append((x, y, bw, bh, score, kps))

        # Group photos: re-run the short-range detector on four overlapping
        # crops so faces that are small in the full frame get a second chance.
        if not _tile and min(h, w) >= 640 and len(kept) < 3:
            tw, th = int(w * 0.60), int(h * 0.60)
            for ox0, oy0 in ((0, 0), (w - tw, 0), (0, h - th), (w - tw, h - th)):
                tile = rgb[oy0 : oy0 + th, ox0 : ox0 + tw]
                if tile.size == 0:
                    continue
                for x, y, bw, bh, score, kps in self._blaze_detect(tile, _tile=True):
                    mapped = [(kx + ox0, ky + oy0) for kx, ky in kps] if kps else None
                    kept.append((x + ox0, y + oy0, bw, bh, score, mapped))
            # _merge_boxes() returns (box, landmarks) pairs, but callers of
            # _blaze_detect() expect full 6-tuples.  Re-expand them; dropping
            # this conversion used to crash large photos with 1-2 faces with
            # "IndexError: tuple index out of range".
            merged = self._merge_boxes(kept)
            kept = [
                (box[0], box[1], box[2], box[3], 0.90, landmarks)
                for box, landmarks in merged
            ]
        return kept[:20]

    def _detect_faces(
        self, rgb: np.ndarray
    ) -> list[tuple[tuple[int, int, int, int], list[tuple[float, float]] | None]]:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        min_size = max(30, int(min(h, w) * 0.035))
        boxes: list[tuple[int, int, int, int, float, list[tuple[float, float]] | None]] = []
        blaze_boxes = self._blaze_detect(rgb)
        boxes.extend(blaze_boxes)

        def haar_plausible(x: int, y: int, fw: int, fh: int) -> bool:
            """Reject obvious Haar false positives when BlazeFace already has a face.

            The profile cascade happily fires on an arm or a belt buckle.  A
            real face at this size should show either an eye-like dark region
            or a meaningful skin area; otherwise it is only kept when there is
            no BlazeFace result at all (the important Haar fallback case).
            """
            if fw < 64 or fh < 64:
                return False
            crop = bgr[max(0, y) : min(h, y + fh), max(0, x) : min(w, x + fw)]
            if crop.size == 0:
                return False
            try:
                face_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if self._detect_eyes(face_gray):
                    return True
                skin = float(self._skin_mask(crop).mean())
                return skin >= 0.32 and min(fw, fh) >= 72
            except cv2.error:
                return False

        def collect(cascade: cv2.CascadeClassifier, image: np.ndarray, mirrored: bool = False, min_weight: float = 0.85) -> None:
            if cascade.empty():
                return
            try:
                found, _, weights = cascade.detectMultiScale3(
                    image,
                    scaleFactor=1.08,
                    minNeighbors=5,
                    minSize=(min_size, min_size),
                    maxSize=(int(w * 0.95), int(h * 0.95)),
                    outputRejectLevels=True,
                )
            except cv2.error:
                found = cascade.detectMultiScale(
                    image,
                    scaleFactor=1.08,
                    minNeighbors=6,
                    minSize=(min_size, min_size),
                    maxSize=(int(w * 0.95), int(h * 0.95)),
                )
                weights = np.ones(len(found))
            for (fx, fy, fw, fh), weight in zip(found, np.asarray(weights).ravel()):
                # Haar's level weight separates a real face from a face-shaped
                # patch of background far better than the box alone.
                if float(weight) < min_weight:
                    continue
                x = w - int(fx) - int(fw) if mirrored else int(fx)
                box = (x, int(fy), int(fw), int(fh))
                if blaze_boxes:
                    overlaps_blaze = any(self._iou(box, b[:4]) > 0.25 for b in blaze_boxes)
                    if not overlaps_blaze and not haar_plausible(x, int(fy), int(fw), int(fh)):
                        continue
                boxes.append((x, int(fy), int(fw), int(fh), float(weight), None))

        front, profile, _eye, _eye_glasses = self._cascades()
        # Plain grayscale is far less prone to face-shaped background false
        # positives than an equalised image.  Only fall back to equalisation
        # when neither BlazeFace nor the plain pass found anything.
        collect(front, gray)
        collect(profile, gray)
        collect(profile, cv2.flip(gray, 1), mirrored=True)
        if not boxes:
            gray_eq = cv2.equalizeHist(gray)
            collect(front, gray_eq, min_weight=1.15)
            collect(profile, gray_eq, min_weight=1.15)
            collect(profile, cv2.flip(gray_eq, 1), mirrored=True, min_weight=1.15)
        return self._merge_boxes(boxes)

    @staticmethod
    def _skin_mask(bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        m1 = (cr >= 130) & (cr <= 182) & (cb >= 72) & (cb <= 138) & (y > 38)
        b, g, r = cv2.split(bgr.astype(np.int16))
        m2 = (r > 55) & (g > 25) & (b > 12) & (r > g) & (r > b) & ((r - np.minimum(g, b)) > 8)
        m = (m1 | m2).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return m

    def _detect_eyes(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = gray.shape
        upper = gray[: int(h * 0.66), :]
        min_side = max(8, int(w * 0.10))
        found: list[tuple[int, int, int, int]] = []
        _front, _profile, eye, eye_glasses = self._cascades()
        for cascade in (eye, eye_glasses):
            if cascade.empty():
                continue
            dets = cascade.detectMultiScale(
                upper, scaleFactor=1.05, minNeighbors=3, minSize=(min_side, min_side)
            )
            for x, y, ew, eh in dets:
                if ew > w * 0.42 or eh > h * 0.32:
                    continue
                if y > h * 0.54:
                    continue
                box = (int(x), int(y), int(ew), int(eh))
                if any(self._iou(box, k) > 0.4 for k in found):
                    continue
                found.append(box)
        return found[:2]

    def _face_quality(self, face_bgr: np.ndarray) -> tuple[float, dict[str, float]]:
        interp = cv2.INTER_AREA if max(face_bgr.shape[:2]) > FACE_NORMAL_SIZE else cv2.INTER_CUBIC
        norm = cv2.resize(face_bgr, (FACE_NORMAL_SIZE, FACE_NORMAL_SIZE), interpolation=interp)
        gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_var = float(lap.var())
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        ten = float(np.mean(cv2.magnitude(gx, gy)))
        hf = gray - cv2.GaussianBlur(gray, (0, 0), 1.1)
        hf_ratio = float(hf.std() / (gray.std() + 1e-6))
        score = 100.0 * _clip((_log10(lap_var) - 0.40) / 1.90)
        return round(score, 1), {
            "lap_var": round(lap_var, 2),
            "tenengrad": round(ten, 2),
            "hf_ratio": round(hf_ratio, 3),
        }

    def _occlusion(self, face_bgr: np.ndarray, eyes: list[tuple[int, int, int, int]], face_score: float) -> dict[str, Any]:
        h, w = face_bgr.shape[:2]
        skin = self._skin_mask(face_bgr)
        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)

        def region(xa: float, xb: float, ya: float, yb: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            x0, x1 = int(w * xa), int(w * xb)
            y0, y1 = int(h * ya), int(h * yb)
            return skin[y0:y1, x0:x1], hsv[y0:y1, x0:x1], lap[y0:y1, x0:x1]

        def ratio(xa: float, xb: float, ya: float, yb: float) -> float:
            m, _, _ = region(xa, xb, ya, yb)
            return float(m.mean()) if m.size else 0.0

        def edge_density(xa: float, xb: float, ya: float, yb: float) -> float:
            _, _, lp = region(xa, xb, ya, yb)
            return float(np.mean(np.abs(lp) > 22)) if lp.size else 0.0

        forehead = ratio(0.24, 0.76, 0.08, 0.27)
        cheek = (ratio(0.07, 0.31, 0.54, 0.75) + ratio(0.69, 0.93, 0.54, 0.75)) / 2.0
        lower = ratio(0.21, 0.79, 0.58, 0.84)
        chin = ratio(0.32, 0.68, 0.80, 0.96)
        eye_band_skin = ratio(0.10, 0.90, 0.29, 0.51)
        eye_left_skin = ratio(0.20, 0.44, 0.33, 0.48)
        eye_right_skin = ratio(0.56, 0.80, 0.33, 0.48)
        eye_skin = min(eye_left_skin, eye_right_skin)
        _, eye_hsv, _ = region(0.10, 0.90, 0.29, 0.51)
        eye_dark = float(np.mean(eye_hsv[:, :, 2] < 72)) if eye_hsv.size else 0.0
        eye_v = float(eye_hsv[:, :, 2].mean()) if eye_hsv.size else 0.0
        eye_sat = float(eye_hsv[:, :, 1].mean()) if eye_hsv.size else 0.0
        lower_edge = edge_density(0.30, 0.70, 0.62, 0.82)
        lower_v = float(hsv[int(h * 0.58) : int(h * 0.84), int(w * 0.21) : int(w * 0.79), 2].mean())

        ref_skin = max(forehead, cheek)
        reasons: list[str] = []
        notes: list[str] = []

        mask_score = 0.0
        if lower < 0.50 and ref_skin > 0.42:
            mask_score = _clip((0.60 - lower) / 0.38) * _clip(ref_skin / 0.62)
            if lower_edge > 0.30 or (lower_v < 62 and lower_edge > 0.20):
                mask_score *= 0.42
                notes.append("下半脸纹理较强，可能为胡须/阴影")
            if mask_score >= 0.32:
                reasons.append("下半脸肤色区域明显减少")

        sun_score = 0.0
        if len(eyes) == 0 and face_score >= 35:
            sun_score = (
                _clip((0.55 - eye_skin) / 0.40)
                * _clip((eye_dark - 0.34) / 0.40)
                * _clip((110.0 - eye_v) / 55.0)
            )
            if eye_sat > 70 and eye_v > 95 and eye_dark < 0.45:
                sun_score *= 0.60
                notes.append("眼部区域偏亮/偏彩，可能为反光眼镜")
            if sun_score >= 0.30:
                reasons.append("双眼未检出且眼部区域偏暗")

        occ_score = max(mask_score, sun_score)
        kind = "none"
        if sun_score >= mask_score and sun_score >= 0.30:
            kind = "eyes"
        elif mask_score >= 0.30:
            kind = "lower"

        if occ_score >= 0.48:
            label = {
                "eyes": "眼部疑似遮挡",
                "lower": "下半脸疑似遮挡",
                "none": "面部疑似遮挡",
            }[kind]
        elif occ_score >= 0.30:
            label = {
                "eyes": "眼部特征异常",
                "lower": "下半脸特征异常",
                "none": "面部特征异常",
            }[kind]
        else:
            label = "未见明显遮挡"

        if face_score < 35:
            label = "人脸较虚，遮挡无法判断"

        reason_text = "；".join(reasons) if reasons else (notes[0] if notes else "")
        return {
            "label": label,
            "score": round(float(occ_score), 3),
            "occluded": bool(occ_score >= 0.48),
            "kind": kind,
            "reason": reason_text,
            "reasons": reasons,
            "notes": notes,
            "metrics": {
                "forehead_skin": round(forehead, 3),
                "cheek_skin": round(cheek, 3),
                "lower_skin": round(lower, 3),
                "chin_skin": round(chin, 3),
                "eye_skin": round(eye_skin, 3),
                "eye_band_skin": round(eye_band_skin, 3),
                "eye_dark": round(eye_dark, 3),
                "eye_v": round(eye_v, 1),
                "lower_edge": round(lower_edge, 3),
                "eyes_found": len(eyes),
            },
        }

    @staticmethod
    def _warp_face(bgr: np.ndarray, landmarks: list[tuple[float, float]]) -> np.ndarray | None:
        """Normalize a face with the BlazeFace eye/mouth landmarks.

        The warped 160x160 patch has the eyes at (0.32, 0.40)/(0.68, 0.40)
        and the mouth at (0.50, 0.72), which makes the skin-region checks
        independent of the detector box and of head roll.
        """
        if len(landmarks) < 4:
            return None
        eye_a, eye_b, mouth = np.asarray(landmarks[0], np.float32), np.asarray(landmarks[1], np.float32), np.asarray(landmarks[3], np.float32)
        if eye_a[0] > eye_b[0]:
            eye_a, eye_b = eye_b, eye_a
        eye_dist = float(np.linalg.norm(eye_b - eye_a))
        if eye_dist < 10.0:
            return None
        if mouth[1] < (eye_a[1] + eye_b[1]) / 2.0 - eye_dist * 0.1:
            return None
        src = np.asarray([eye_a, eye_b, mouth], np.float32)
        dst = np.asarray(
            [
                [0.32 * FACE_NORMAL_SIZE, 0.40 * FACE_NORMAL_SIZE],
                [0.68 * FACE_NORMAL_SIZE, 0.40 * FACE_NORMAL_SIZE],
                [0.50 * FACE_NORMAL_SIZE, 0.72 * FACE_NORMAL_SIZE],
            ],
            np.float32,
        )
        try:
            matrix = cv2.getAffineTransform(src, dst)
            return cv2.warpAffine(
                bgr,
                matrix,
                (FACE_NORMAL_SIZE, FACE_NORMAL_SIZE),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except cv2.error:
            return None

    def _analyze_faces(self, rgb: np.ndarray, want_occlusion: bool) -> tuple[list[dict[str, Any]], float | None, list[float]]:
        boxes = self._detect_faces(rgb)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        faces: list[dict[str, Any]] = []
        reliable: list[float] = []
        all_scores: list[float] = []

        for idx, ((x, y, fw, fh), landmarks) in enumerate(boxes):
            # A hair of padding keeps the chin and the sides of the face.
            pad = int(round(max(fw, fh) * 0.04))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + fw + pad), min(h, y + fh + pad)
            crop = bgr[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            warped = self._warp_face(bgr, landmarks) if landmarks else None
            analysis_crop = warped if warped is not None else crop
            score, qmetrics = self._face_quality(analysis_crop)
            gray_crop = cv2.cvtColor(analysis_crop, cv2.COLOR_BGR2GRAY)
            eyes = self._detect_eyes(gray_crop) if want_occlusion else []
            occ = (
                self._occlusion(analysis_crop, eyes, score)
                if want_occlusion
                else {
                    "label": "未启用遮挡分析",
                    "score": 0.0,
                    "occluded": False,
                    "kind": "none",
                    "reasons": [],
                    "notes": [],
                    "metrics": {},
                }
            )
            small = fw < 64 or fh < 64
            if not small:
                reliable.append(score)
            all_scores.append(score)
            faces.append(
                {
                    "box": [round(x0 / w, 5), round(y0 / h, 5), round((x1 - x0) / w, 5), round((y1 - y0) / h, 5)],
                    "pixel_box": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                    "width": int(fw),
                    "score": score,
                    "small": bool(small),
                    "has_landmarks": bool(landmarks),
                    "warped": warped is not None,
                    "quality": qmetrics,
                    "eyes": len(eyes),
                    "occlusion": occ,
                }
            )

        face_score: float | None = None
        if reliable:
            vals = sorted(reliable)
            base = float(sum(vals) / len(vals))
            if len(vals) >= 2 and vals[0] < base * 0.58:
                base = 0.62 * vals[0] + 0.38 * base
            face_score = round(base, 1)
        return faces, face_score, all_scores

    # --------------------------------------------------------------- main work
    def _analyze_uncached(self, path: Path, options: dict[str, Any]) -> dict[str, Any]:
        probe = probe_image(path, deep=False)
        if probe.get("code") != "ok":
            return make_error_result(path, probe)
        try:
            rgb, orig_w, orig_h = self._open_analysis_rgb(path)
        except Exception as exc:  # noqa: BLE001 - every failure becomes a readable card
            return make_error_result(path, describe_exception(exc, path))

        try:
            metrics = self._global_metrics(rgb)
            faces: list[dict[str, Any]] = []
            face_score: float | None = None
            if options.get("faces", True):
                faces, face_score, _ = self._analyze_faces(rgb, bool(options.get("occlusion", True)))

            if face_score is not None:
                score = 0.72 * face_score + 0.28 * metrics["global_score"]
                confidence = max(0.78, metrics["confidence"])
            else:
                score = metrics["global_score"]
                confidence = metrics["confidence"]

            occ_faces = [f for f in faces if f["occlusion"].get("occluded")]
            occ_reasons: list[str] = []
            for f in occ_faces:
                for r in f["occlusion"].get("reasons", []):
                    if r not in occ_reasons:
                        occ_reasons.append(r)
            if face_score is not None and faces:
                confidence = max(confidence, 0.8)

            focus = dict(metrics.get("focus") or {"x": 0.5, "y": 0.5, "source": "sharpness",
                                                   "confidence": 0.0, "label": "画面细节"})
            reliable_faces = [(i + 1, f) for i, f in enumerate(faces)
                              if f.get("width", 0) >= 64 and f.get("score", 0) >= 35]
            # BlazeFace keypoints are a much stronger signal than a Haar-only
            # box; prefer those for the estimated focus, then sharpness/size.
            with_landmarks = [(i, f) for i, f in reliable_faces if f.get("has_landmarks")]
            if with_landmarks:
                reliable_faces = with_landmarks
            if reliable_faces:
                face_no, best = max(reliable_faces, key=lambda item: (item[1].get("score", 0), item[1].get("width", 0)))
                bx, by, bw, bh = best["box"]
                face_conf = _clip(best.get("score", 0) / 100.0)
                focus = {
                    "x": round(bx + bw / 2, 4),
                    "y": round(by + bh * 0.52, 4),
                    "source": "face",
                    "confidence": round(face_conf, 3),
                    "label": f"人脸 {face_no}",
                }

            thumb = Image.fromarray(rgb)
            thumb.thumbnail((480, 480), Image.Resampling.LANCZOS)
            thumb_rgb = np.asarray(thumb, dtype=np.uint8).copy()
            fp_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            fp_gray = cv2.resize(fp_gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
            fgx = cv2.Sobel(fp_gray, cv2.CV_32F, 1, 0, ksize=3)
            fgy = cv2.Sobel(fp_gray, cv2.CV_32F, 0, 1, ksize=3)
            fp_edges = cv2.resize(cv2.magnitude(fgx, fgy), (32, 32), interpolation=cv2.INTER_AREA)
            fingerprint = base64.b64encode(fp_edges.astype(np.uint8).tobytes()).decode("ascii")
        except Exception as exc:  # noqa: BLE001
            import traceback

            result = make_error_result(path, describe_exception(exc, path), orig_w, orig_h)
            result["error_trace"] = traceback.format_exc()
            return result

        return {
            "version": ANALYSIS_VERSION,
            "path": str(path),
            "error": None,
            "width": orig_w,
            "height": orig_h,
            "score": round(float(_clip(score, 0, 100)), 1),
            "global_score": metrics["global_score"],
            "face_score": face_score,
            "faces": faces,
            "face_count": len(faces),
            "occluded": bool(occ_faces),
            "occlusion_reasons": occ_reasons,
            "confidence": round(float(_clip(confidence)), 3),
            "low_detail": metrics["low_detail"] and face_score is None,
            "focus": focus,
            "metrics": {k: v for k, v in metrics.items() if k not in ("global_score", "confidence", "low_detail", "focus")},
            "phash": metrics.get("phash"),
            "fingerprint": fingerprint,
            "_thumb_rgb": thumb_rgb,
        }


def _decode_fingerprint(value: Any) -> np.ndarray | None:
    if not value:
        return None
    try:
        arr = np.frombuffer(base64.b64decode(value), dtype=np.uint8)
        if arr.size != 32 * 32:
            return None
        return arr.reshape(32, 32)
    except Exception:
        return None


def _phash_distance(a: Any, b: Any) -> int:
    try:
        return (int(str(a), 16) ^ int(str(b), 16)).bit_count()
    except (TypeError, ValueError):
        return 64


def find_similar_groups(records: list[dict[str, Any]]) -> list[list[str]]:
    """Group visibly identical photos across formats/resolutions.

    ``records`` items need ``key`` and the analysis fields ``fingerprint``,
    ``phash``, ``width``/``height``.  The comparison is intentionally strict:
    a missed group is much safer than folding a blurry or cropped version into
    a group that a bulk-delete action could then remove.
    """
    items: list[tuple[str, dict[str, Any], np.ndarray]] = []
    for record in records:
        if record.get("error") or not record.get("fingerprint"):
            continue
        fp = _decode_fingerprint(record.get("fingerprint"))
        if fp is None:
            continue
        key = str(record.get("key", ""))
        width, height = _safe_float(record.get("width")), _safe_float(record.get("height"))
        if not key or width <= 0 or height <= 0:
            continue
        items.append((key, record, fp))

    parent = {key: key for key, _, _ in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, (key_a, a, fp_a) in enumerate(items):
        for key_b, b, fp_b in items[i + 1 :]:
            wa, ha = _safe_float(a.get("width")), _safe_float(a.get("height"))
            wb, hb = _safe_float(b.get("width")), _safe_float(b.get("height"))
            if min(wa, ha, wb, hb) <= 0:
                continue
            # Different crops change the aspect ratio; reject those.
            if abs((wa / ha) - (wb / hb)) > 0.035:
                continue
            if _phash_distance(a.get("phash"), b.get("phash")) > 18:
                continue
            diff = np.abs(fp_a.astype(np.int16) - fp_b.astype(np.int16))
            mean_diff = float(diff.mean())
            strong_diff = float((diff > 25).mean())
            # Edge fingerprints change when a version is blurred, cropped or
            # retouched; both global and local agreement are required.
            if mean_diff <= 2.8 and strong_diff <= 0.004:
                union(key_a, key_b)

    groups: dict[str, list[str]] = {}
    for key, _, _ in items:
        groups.setdefault(find(key), []).append(key)
    return [group for group in groups.values() if len(group) > 1]


def status_for(result: dict[str, Any], threshold: float) -> str:
    """Return pending / error / sharp / review / blur / uncertain."""
    if not result:
        return "pending"
    if result.get("error"):
        return "error"
    if result.get("status") == "pending":
        return "pending"
    score = _safe_float(result.get("score"))
    if result.get("low_detail") and score < threshold + 8:
        return "uncertain"
    if score >= threshold:
        return "sharp"
    if score >= threshold - 15:
        return "review"
    return "blur"


STATUS_LABELS = {
    "pending": "分析中",
    "error": "无法读取",
    "sharp": "清晰",
    "review": "待复核",
    "blur": "模糊",
    "uncertain": "细节少",
}

STATUS_COLORS = {
    "pending": "#7d8699",
    "error": "#fb7185",
    "sharp": "#5ee0a0",
    "review": "#f6c45b",
    "blur": "#ff7185",
    "uncertain": "#8fa6c8",
}
