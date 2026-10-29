"""FocusSort - a local photo culling desk for spotting soft / out-of-focus shots.

Run with:  pythonw app.py
Everything stays on this computer.  The only optional network use is the
initial dependency installation; image analysis is fully offline.
"""
from __future__ import annotations

import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from ctypes import wintypes
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk

from analyzer import (
    BLAZE_MODEL,
    ERROR_ACTIONS,
    IMAGE_EXTS,
    Analyzer,
    find_similar_groups,
    make_error_result,
    probe_image,
    status_for,
)
from theme import (
    ACCENT,
    ACCENT_HOVER,
    AMBER,
    BG,
    BLACK,
    BLUE,
    BORDER,
    BORDER_LIGHT,
    CARD,
    CARD_HOVER,
    CARD_PRESSED,
    CORAL,
    FAINT,
    MINT,
    MUTED,
    PANEL,
    PANEL_2,
    STATUS_COLORS,
    STATUS_LABELS,
    TEXT,
    TEXT_SOFT,
    VIOLET,
    WHITE,
    Fonts,
    IconButton,
    PillButton,
    Segmented,
    ThresholdSlider,
    ToggleSwitch,
    ToolTip,
    ellipsize,
    human_size,
    natural_key,
    round_rect,
)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False


APP_NAME = "FocusSort"
APP_TITLE = "焦点 FocusSort · 照片清晰度快筛"
APP_VERSION = "1.5"
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("LOCALAPPDATA") or Path.home()) / "FocusSort"
CACHE_DIR = DATA_DIR / "cache"
SETTINGS_PATH = DATA_DIR / "settings.json"
DEFAULT_THRESHOLD = 60.0
MAX_RESTORE_FILES = 300
DEFAULT_CACHE_DAYS = 30

DEFAULT_SETTINGS: dict[str, Any] = {
    "settings_version": 2,
    "threshold": DEFAULT_THRESHOLD,
    "sort": "import",
    "filter": "all",
    "geometry": "1560x960",
    "recursive": True,
    "last_dir": str(Path.home() / "Pictures"),
    "preview_heatmap": False,
    # Overlays are opt-in: the photo itself is never covered on first launch.
    "preview_faces": False,
    # Focus is an inference, not camera AF data.  Covering the photo by default
    # was the main viewing regression; keep it off until explicitly requested.
    "preview_focus": False,
    "restore_session": True,
    "cache_days": DEFAULT_CACHE_DAYS,
    "last_files": [],
}


def ensure_data_dir() -> None:
    """Create a writable data folder, falling back to the app folder/temp."""
    global DATA_DIR, CACHE_DIR, SETTINGS_PATH
    candidates = [
        Path(os.getenv("LOCALAPPDATA") or Path.home()) / "FocusSort",
        APP_DIR / ".data",
        Path(os.getenv("TEMP") or ".") / "FocusSort",
    ]
    for candidate in candidates:
        try:
            (candidate / "cache" / "meta").mkdir(parents=True, exist_ok=True)
            (candidate / "cache" / "thumbs").mkdir(parents=True, exist_ok=True)
            DATA_DIR = candidate
            CACHE_DIR = candidate / "cache"
            SETTINGS_PATH = candidate / "settings.json"
            return
        except OSError:
            continue


# ---------------------------------------------------------------------------
# platform niceties
# ---------------------------------------------------------------------------
def enable_dpi_awareness() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def enable_dark_titlebar(root: tk.Misc) -> None:
    try:
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        value = ctypes.c_int(1)
        for attr in (20, 19):
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(value), ctypes.sizeof(value))
    except Exception:
        pass


def recycle_files(paths: list[Path], hwnd: int = 0) -> tuple[bool, str]:
    """Move files to the Windows Recycle Bin (never permanent-delete)."""
    existing = [str(p) for p in paths if p.exists()]
    if not existing:
        return False, "文件不存在"
    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR),
            ("fFlags", wintypes.WORD),
            ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", wintypes.LPCWSTR),
        ]
    FO_DELETE = 3
    FOF_SILENT = 0x0004
    FOF_NOCONFIRMATION = 0x0010
    FOF_ALLOWUNDO = 0x0040
    FOF_NOERRORUI = 0x0400
    FOF_WANTNUKEWARNING = 0x4000
    raw = "\0".join(existing) + "\0\0"
    buf = ctypes.create_unicode_buffer(raw)
    op = SHFILEOPSTRUCTW()
    op.hwnd = hwnd
    op.wFunc = FO_DELETE
    op.pFrom = ctypes.cast(buf, wintypes.LPCWSTR)
    op.fFlags = FOF_ALLOWUNDO | FOF_NOERRORUI | FOF_WANTNUKEWARNING
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if result != 0:
        return False, f"回收站操作失败（错误码 {result}）"
    if op.fAnyOperationsAborted:
        return False, "操作已取消"
    return True, ""


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclass
class Photo:
    path: Path
    key: str
    name: str
    mtime: float
    size: int
    result: dict[str, Any] | None = None
    state: str = "pending"  # pending / done / error
    cache_key: str = ""
    thumb_path: Path | None = None
    added_at: float = field(default_factory=time.time)

    def status(self, threshold: float) -> str:
        if self.state == "error":
            return "error"
        if self.state == "cancelled":
            return "cancelled"
        if self.result is None:
            return "pending"
        return status_for(self.result, threshold)

    @property
    def is_error(self) -> bool:
        return self.state == "error" or bool(self.result and self.result.get("error"))

    def error_reason(self) -> str:
        if not self.result:
            return "无法读取图片"
        return str(self.result.get("error") or self.result.get("error_hint") or "无法读取图片")

    def error_hint(self) -> str:
        if not self.result:
            return "可点「重试」重新读取，或从列表移除。"
        return str(
            self.result.get("error_hint")
            or ERROR_ACTIONS.get(self.result.get("error_code", ""), "")
            or "可点「重试」重新读取；如果文件已移动，请「从列表移除」后重新导入。"
        )


def load_settings() -> dict[str, Any]:
    defaults = dict(DEFAULT_SETTINGS)
    loaded_version: Any = None
    try:
        data = json.loads(SETTINGS_PATH.read_text("utf-8"))
        if isinstance(data, dict):
            loaded_version = data.get("settings_version")
            defaults.update({k: v for k, v in data.items() if k in defaults})
    except (OSError, ValueError):
        pass
    # Defensive migration for settings written by older versions.
    if not isinstance(defaults.get("last_files"), list):
        defaults["last_files"] = []
    defaults["last_files"] = [str(x) for x in defaults["last_files"] if isinstance(x, (str, os.PathLike))]
    try:
        old_version = int(loaded_version) if loaded_version is not None else 1
    except (TypeError, ValueError):
        old_version = 1
    if old_version < 2:
        # v1.4 could leave overlays on.  v1.5 starts with a clean photo view,
        # especially the old always-on large gold focus crosshair.
        defaults["preview_heatmap"] = False
        defaults["preview_faces"] = False
        defaults["preview_focus"] = False
    defaults["settings_version"] = 2
    defaults["preview_focus"] = bool(defaults.get("preview_focus", False))
    defaults["restore_session"] = bool(defaults.get("restore_session", True))
    return defaults


def save_settings(data: dict[str, Any]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        os.replace(tmp, SETTINGS_PATH)
    except OSError:
        pass


def log_error(text: str) -> None:
    """Append a user-diagnosable traceback without ever failing the UI."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / "error.log"
        if path.exists() and path.stat().st_size > 2_000_000:
            try:
                path.unlink()
            except OSError:
                pass
        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n--- {datetime.now().isoformat(timespec='seconds')} ---\n")
            f.write(text.rstrip() + "\n")
    except OSError:
        pass


def collect_image_paths(paths: list[Path], recursive: bool = True) -> tuple[list[Path], int]:
    """Return unique image files and the number of ignored non-image files."""
    out: list[Path] = []
    seen: set[str] = set()
    ignored = 0
    skip_dirs = {"$recycle.bin", "system volume information", ".git", ".svn", "__pycache__", "node_modules"}

    def add_file(p: Path) -> None:
        nonlocal ignored
        if p.suffix.lower() not in IMAGE_EXTS:
            ignored += 1
            return
        try:
            key = os.path.normcase(str(p.resolve()))
        except OSError:
            key = os.path.normcase(str(p))
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    for raw in paths:
        try:
            p = Path(raw).expanduser()
        except (TypeError, ValueError):
            ignored += 1
            continue
        if p.is_file():
            add_file(p)
        elif p.is_dir():
            if recursive:
                for root, dirs, files in os.walk(p, topdown=True):
                    dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() not in skip_dirs]
                    for name in files:
                        if name.startswith("."):
                            ignored += 1
                            continue
                        add_file(Path(root) / name)
            else:
                try:
                    for child in p.iterdir():
                        if child.is_file():
                            add_file(child)
                except OSError:
                    ignored += 1
        else:
            ignored += 1
    return out, ignored


def collect_import_candidates(paths: list[Path], recursive: bool = True) -> tuple[list[Path], int, list[Path]]:
    """Expand an import gesture while keeping explicitly bad files visible.

    Returns ``(supported_files, ignored_count, explicit_bad_files)``.  Only
    files/directories the user explicitly pointed at can become "bad" cards;
    ordinary non-image files inside a dropped folder are just ignored.
    """
    files: list[Path] = []
    seen: set[str] = set()
    bad: list[Path] = []
    ignored = 0

    def key_for(p: Path) -> str:
        try:
            return os.path.normcase(str(p.resolve()))
        except OSError:
            return os.path.normcase(os.path.abspath(str(p)))

    def add(p: Path, explicit: bool) -> None:
        nonlocal ignored
        key = key_for(p)
        if key in seen:
            return
        seen.add(key)
        try:
            exists = p.is_file()
        except OSError:
            exists = False
        if p.suffix.lower() in IMAGE_EXTS and exists:
            files.append(p)
        elif explicit:
            # Explicitly chosen files remain visible as error cards even when
            # they are unsupported or have already been moved/deleted.
            bad.append(p)
        else:
            ignored += 1

    for raw in paths:
        try:
            p = Path(raw).expanduser()
        except (TypeError, ValueError):
            ignored += 1
            continue
        try:
            is_dir = p.is_dir()
            is_file = p.is_file()
        except OSError:
            is_dir = is_file = False
        if is_dir:
            found, skipped = collect_image_paths([p], recursive=recursive)
            ignored += skipped
            for f in found:
                add(f, explicit=False)
        elif is_file:
            add(p, explicit=True)
        elif not p.exists() and p.suffix.lower() in IMAGE_EXTS:
            # Missing explicit photo: show an actionable error card rather
            # than silently dropping it.
            add(p, explicit=True)
        else:
            ignored += 1
    return files, ignored, bad


def validate_restore_candidates(items: list[str] | list[Path], cap: int = MAX_RESTORE_FILES) -> tuple[list[Path], int, int]:
    """Validate a previous-session path list without touching the files.

    Returns ``(valid_paths, invalid_count, limited_count)``.  Invalid means
    missing, unreadable, empty, unsupported or header-corrupt.  Once ``cap``
    valid photos are found, remaining entries are counted as limited so a huge
    last-session list cannot stall startup.
    """
    valid: list[Path] = []
    invalid = 0
    limited = 0
    seen: set[str] = set()
    for item in items:
        try:
            p = Path(item)
        except (TypeError, ValueError):
            invalid += 1
            continue
        try:
            key = os.path.normcase(str(p.resolve()))
        except OSError:
            key = os.path.normcase(os.path.abspath(str(p)))
        if key in seen:
            continue
        seen.add(key)
        if len(valid) >= cap:
            # Beyond the cap we do not decode headers, but still distinguish a
            # genuinely dead entry from a valid one that was skipped.
            if probe_image(p, deep=False).get("code") == "ok":
                limited += 1
            else:
                invalid += 1
            continue
        if probe_image(p, deep=True).get("code") == "ok":
            valid.append(p)
        else:
            invalid += 1
    return valid, invalid, limited


def load_full_and_preview(path: Path) -> tuple[Image.Image, Image.Image]:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        if max(im.size) > 7000:
            im.thumbnail((7000, 7000), Image.Resampling.LANCZOS)
        full = im.copy()
        preview = im.copy()
        preview.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    return full, preview


def load_card_source(path: Path, thumb_path: Path | None) -> Image.Image:
    if thumb_path and thumb_path.exists():
        try:
            with Image.open(thumb_path) as im:
                return im.convert("RGB")
        except Exception:
            try:
                thumb_path.unlink()
            except OSError:
                pass
    with Image.open(path) as im:
        im.draft("RGB", (900, 900))
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.thumbnail((480, 480), Image.Resampling.LANCZOS)
        return im.copy()


def rounded_image(img: Image.Image, radius: int = 10) -> Image.Image:
    img = img.convert("RGBA")
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, img.size[0] - 1, img.size[1] - 1), radius=radius, fill=255)
    img.putalpha(mask)
    return img


def fit_on_card(img: Image.Image, size: tuple[int, int], bg: str = CARD) -> Image.Image:
    w, h = size
    img = img.convert("RGB")
    scale = min(w / img.width, h / img.height)
    nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(resized, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def format_date(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except (OSError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# toast
# ---------------------------------------------------------------------------
class Toast:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.frame: tk.Frame | None = None
        self.job = None

    def show(self, text: str, ms: int = 2600) -> None:
        self.hide()
        frame = tk.Frame(self.root, bg="#232a38", highlightthickness=1, highlightbackground=BORDER_LIGHT)
        label = tk.Label(
            frame,
            text=text,
            bg="#232a38",
            fg=TEXT,
            font=Fonts(self.root).small,
            padx=16,
            pady=9,
        )
        label.pack()
        frame.place(relx=0.5, rely=1.0, y=-54, anchor="s")
        frame.lift()
        self.frame = frame
        self.job = self.root.after(ms, self.hide)

    def hide(self) -> None:
        if self.job:
            try:
                self.root.after_cancel(self.job)
            except tk.TclError:
                pass
            self.job = None
        if self.frame:
            try:
                self.frame.destroy()
            except tk.TclError:
                pass
            self.frame = None


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
class PhotoGrid(tk.Frame):
    CARD_W = 224
    CARD_H = 194
    GAP = 18

    def __init__(self, master, app: "FocusSortApp"):
        super().__init__(master, bg=BG)
        self.app = app
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0, yscrollincrement=18)
        self.scrollbar = tk.Canvas(self, width=8, bg=BG, highlightthickness=0, bd=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y", padx=(2, 4), pady=8)
        self.canvas.configure(yscrollcommand=self._on_scrollbar)
        self._scroll_thumb = None
        self._scroll_drag = 0
        self._hover_id: str | None = None
        self._hitboxes: list[tuple[float, float, float, float, str]] = []
        self._render_job = None
        self._visible_count = 0
        self._cols = 4
        self._thumb_refs: OrderedDict[tuple[str, int, int], ImageTk.PhotoImage] = OrderedDict()
        self._empty_button: tuple[float, float, float, float] | None = None
        self._empty_action = "add"
        self._drop_active = False

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Double-Button-1>", self._on_double)
        self.canvas.bind("<Button-3>", self._on_right)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        self.scrollbar.bind("<Button-1>", self._scroll_click)
        self.scrollbar.bind("<B1-Motion>", self._scroll_drag_to)
        self.scrollbar.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_scroll_drag", 0))

    # ------------------------------------------------------------------ scroll
    def _on_scrollbar(self, first: str, last: str) -> None:
        self._draw_scrollbar(float(first), float(last))

    def _scroll_click(self, event):
        if not self._scroll_thumb:
            return
        y1 = self._scroll_thumb[1]
        total = max(1, self.canvas.bbox("all")[3] if self.canvas.bbox("all") else 1)
        view_h = max(1, self.canvas.winfo_height())
        if event.y < y1:
            self.canvas.yview_scroll(-1, "pages")
        else:
            self.canvas.yview_scroll(1, "pages")
        self._scroll_drag = event.y

    def _scroll_drag_to(self, event):
        if not self._scroll_thumb:
            return
        y1, y2 = self._scroll_thumb[1], self._scroll_thumb[3]
        h = self.scrollbar.winfo_height()
        denom = max(1, h - 8 - (y2 - y1))
        frac = max(0.0, min(1.0, (event.y - 4 - (y2 - y1) / 2) / denom))
        self.canvas.yview_moveto(frac)
        self.schedule_render()

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120) * 3, "units")
        self.schedule_render()

    # ------------------------------------------------------------------ layout
    def _on_configure(self, _event=None):
        self.schedule_render()

    def _grid_metrics(self) -> tuple[int, int, int, int]:
        width = max(1, self.canvas.winfo_width())
        cols = max(1, (width - 2 * 18 + self.GAP) // (self.CARD_W + self.GAP))
        total_w = cols * self.CARD_W + (cols - 1) * self.GAP
        pad_x = max(18, (width - total_w) // 2)
        return cols, pad_x, self.CARD_W, self.CARD_H

    def schedule_render(self) -> None:
        if self._render_job:
            return
        self._render_job = self.after(16, self._render)

    def _render(self) -> None:
        self._render_job = None
        self.canvas.delete("all")
        self._hitboxes = []
        self._empty_button = None
        photos = self.app.visible_photos
        self._visible_count = len(photos)
        self._cols, pad_x, cw, ch = self._grid_metrics()
        gap = self.GAP
        if not photos:
            self._draw_empty()
            self.canvas.configure(scrollregion=(0, 0, max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())))
            self._draw_scrollbar(0, 1)
            return

        rows = (len(photos) + self._cols - 1) // self._cols
        total_h = 18 + rows * ch + max(0, rows - 1) * gap + 26
        self.canvas.configure(scrollregion=(0, 0, max(1, self.canvas.winfo_width()), total_h))
        top = self.canvas.canvasy(0)
        bottom = top + self.canvas.winfo_height()
        first_row = max(0, int((top - 18) // (ch + gap)) - 1)
        last_row = int((bottom - 18) // (ch + gap)) + 2
        first = first_row * self._cols
        last = min(len(photos), (last_row + 1) * self._cols)
        for idx in range(first, last):
            row, col = divmod(idx, self._cols)
            x = pad_x + col * (cw + gap)
            y = 18 + row * (ch + gap)
            self._draw_card(photos[idx], x, y, cw, ch)
        if self._drop_active:
            self.canvas.create_rectangle(8, 8, max(9, self.canvas.winfo_width() - 8), max(9, self.canvas.winfo_height() - 8),
                                         outline=ACCENT, width=2, dash=(7, 5))
        visible = self.canvas.yview()
        self._draw_scrollbar(visible[0], visible[1])

    def _draw_scrollbar(self, first: float, last: float) -> None:
        self.scrollbar.delete("all")
        h = self.scrollbar.winfo_height()
        w = self.scrollbar.winfo_width()
        if h < 20:
            return
        round_rect(self.scrollbar, w / 2 - 3, 4, w / 2 + 3, h - 4, 3, fill="#1b202b", outline="")
        if last - first >= 0.999:
            return
        y1 = 4 + (h - 8) * first
        y2 = 4 + (h - 8) * last
        if y2 - y1 < 28:
            y2 = y1 + 28
        if y2 > h - 4:
            y2 = h - 4
            y1 = max(4, y2 - 28)
        self._scroll_thumb = (4, y1, h - 4, y2)
        round_rect(self.scrollbar, w / 2 - 2.5, y1, w / 2 + 2.5, y2, 2.5, fill="#3a4354", outline="")

    # ------------------------------------------------------------------ cards
    def _thumb_for(self, photo: Photo, w: int, h: int) -> ImageTk.PhotoImage:
        key = (photo.key, w, h)
        cached = self._thumb_refs.get(key)
        if cached is not None:
            self._thumb_refs.move_to_end(key)
            return cached
        try:
            src = load_card_source(photo.path, photo.thumb_path)
            img = fit_on_card(src, (w, h))
            img = rounded_image(img, 9)
            tkimg = ImageTk.PhotoImage(img)
        except Exception:
            img = Image.new("RGB", (w, h), CARD)
            d = ImageDraw.Draw(img)
            d.text((10, 10), "无法预览", fill=CORAL)
            tkimg = ImageTk.PhotoImage(img)
        self._thumb_refs[key] = tkimg
        while len(self._thumb_refs) > 120:
            self._thumb_refs.popitem(last=False)
        return tkimg

    def _draw_card(self, photo: Photo, x: float, y: float, w: float, h: float) -> None:
        app = self.app
        selected = photo.key in app.selected_keys
        hover = photo.key == self._hover_id
        bg = CARD_HOVER if (hover or selected) else CARD
        outline = ACCENT if selected else (BORDER_LIGHT if hover else BORDER)
        width = 2 if selected else 1
        round_rect(self.canvas, x, y, x + w, y + h, 14, fill=bg, outline=outline, width=width)
        inner_w, inner_h = int(w - 20), 132
        ix, iy = x + 10, y + 10
        status = photo.status(app.threshold)

        if photo.is_error:
            self.canvas.create_rectangle(ix, iy, ix + inner_w, iy + inner_h, fill="#19131a", outline="")
            self.canvas.create_text(
                ix + inner_w / 2,
                iy + 29,
                text="!",
                fill=CORAL,
                font=(app.fonts.family, 26, "bold"),
            )
            self.canvas.create_text(
                ix + inner_w / 2,
                iy + 62,
                text="无法读取",
                fill="#e3a3ad",
                font=app.fonts.body_bold,
            )
            reason = ellipsize(photo.error_reason(), app.font_obj(app.fonts.tiny), inner_w - 18)
            self.canvas.create_text(
                ix + inner_w / 2,
                iy + 90,
                text=reason,
                fill="#b88991",
                font=app.fonts.tiny,
                width=inner_w - 18,
                justify="center",
            )
            self.canvas.create_text(
                ix + inner_w / 2,
                iy + inner_h - 18,
                text="选中后用右侧按钮处理",
                fill=FAINT,
                font=app.fonts.tiny,
            )
        elif photo.result is None:
            self.canvas.create_rectangle(ix, iy, ix + inner_w, iy + inner_h, fill="#11141b", outline="")
            if photo.state == "cancelled":
                self.canvas.create_text(
                    ix + inner_w / 2, iy + inner_h / 2 - 8,
                    text="已取消", fill=FAINT, font=app.fonts.body,
                )
                self.canvas.create_text(
                    ix + inner_w / 2, iy + inner_h / 2 + 16,
                    text="可重新分析", fill=FAINT, font=app.fonts.tiny,
                )
            else:
                self.canvas.create_text(
                    ix + inner_w / 2,
                    iy + inner_h / 2 - 8,
                    text="分析中",
                    fill=MUTED,
                    font=app.fonts.body,
                )
                for i in range(3):
                    self.canvas.create_oval(
                        ix + inner_w / 2 - 14 + i * 12,
                        iy + inner_h / 2 + 14,
                        ix + inner_w / 2 - 10 + i * 12,
                        iy + inner_h / 2 + 18,
                        fill=ACCENT if i == 0 else "#414a5d",
                        outline="",
                    )
        else:
            img = self._thumb_for(photo, inner_w, inner_h)
            self.canvas.create_image(ix, iy, image=img, anchor="nw")
            if status == "pending":
                self.canvas.create_rectangle(ix, iy, ix + inner_w, iy + inner_h, fill="#11141b", stipple="gray50", outline="")

        # status pill
        label = STATUS_LABELS.get(status, "分析中")
        color = STATUS_COLORS.get(status, MUTED)
        pill_text = label
        if photo.result and status not in ("error", "pending"):
            pill_text = f"{label}  {photo.result.get('score', 0):.0f}"
        font = app.fonts.small_bold
        pw = app.measure(pill_text, font) + 18
        ph = 23
        px, py = x + 16, y + 16
        round_rect(self.canvas, px, py, px + pw, py + ph, 7, fill=color, outline="")
        self.canvas.create_text(px + pw / 2, py + ph / 2, text=pill_text, fill="#101318", font=font)

        # face / occlusion chips, laid out from the right edge without overlap
        if photo.result:
            chips: list[tuple[str, str, str, int]] = []
            fc = photo.result.get("face_count", 0)
            if fc:
                text = f"{fc} 人"
                chips.append((text, "#11141b", TEXT_SOFT, app.measure(text, app.fonts.tiny) + 14))
            if photo.result.get("occluded"):
                text = "遮挡"
                chips.append((text, "#13253c", BLUE, app.measure(text, app.fonts.tiny) + 14))
            total = sum(c[3] for c in chips) + max(0, len(chips) - 1) * 6
            cx = x + w - 16 - total
            for text, bgc, fgc, fw in chips:
                round_rect(self.canvas, cx, y + 16, cx + fw, y + 39, 7, fill=bgc, outline="")
                self.canvas.create_text(cx + fw / 2, y + 27.5, text=text, fill=fgc, font=app.fonts.tiny)
                cx += fw + 6
        gidx = app.dup_of.get(photo.key)
        if gidx is not None:
            text = f"同图×{len(app.dup_groups[gidx])}"
            fw = app.measure(text, app.fonts.tiny) + 14
            bx, by = x + w - 16 - fw, y + 10 + inner_h - 28
            round_rect(self.canvas, bx, by, bx + fw, by + 22, 7, fill="#2a2140", outline="")
            self.canvas.create_text(bx + fw / 2, by + 11, text=text, fill=VIOLET, font=app.fonts.tiny)

        # text
        name_font = app.fonts.body_bold
        name = ellipsize(photo.name, app.font_obj(name_font), int(w - 32))
        self.canvas.create_text(x + 16, y + 152, text=name, fill=TEXT if not selected else WHITE, font=name_font, anchor="nw")
        if photo.result and not photo.result.get("error"):
            dims = f"{photo.result.get('width', 0)}×{photo.result.get('height', 0)}"
            meta = f"{dims}  ·  {human_size(photo.size)}"
            if photo.result.get("face_count"):
                meta += f"  ·  {photo.result['face_count']} 人"
        elif photo.is_error:
            code = (photo.result or {}).get("error_code", "")
            meta = f"异常 · {code.replace('_', ' ')}" if code else "异常 · 无法读取"
        else:
            meta = "正在计算清晰度与人脸…"
        meta = ellipsize(meta, app.font_obj(app.fonts.tiny), int(w - 32))
        self.canvas.create_text(x + 16, y + 172, text=meta, fill=MUTED, font=app.fonts.tiny, anchor="nw")

        self._hitboxes.append((x, y, x + w, y + h, photo.key))

    # ------------------------------------------------------------------ empty
    def _draw_empty(self) -> None:
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        cx, cy = w / 2, h / 2 - 18
        filtered_empty = bool(self.app.photos) and self._visible_count == 0
        if filtered_empty:
            self.canvas.create_text(cx, cy - 58, text="◌", fill=ACCENT, font=(self.app.fonts.family, 36))
            self.canvas.create_text(cx, cy - 6, text="这个筛选下没有照片", fill=TEXT, font=self.app.fonts.h1)
            self.canvas.create_text(
                cx,
                cy + 28,
                text="试试把清晰判定线调低，或切换到「全部照片」",
                fill=MUTED,
                font=self.app.fonts.small,
            )
            bx, by, bw, bh = cx - 70, cy + 58, 140, 36
            self._empty_button = (bx, by, bx + bw, by + bh)
            self._empty_action = "filter"
            round_rect(self.canvas, bx, by, bx + bw, by + bh, 10, fill=ACCENT, outline="")
            self.canvas.create_text(cx, by + bh / 2, text="显示全部", fill="#1a1309", font=self.app.fonts.body_bold)
            return

        # photo frame icon
        fw, fh = 92, 70
        x1, y1 = cx - fw / 2, cy - 100
        round_rect(self.canvas, x1, y1, x1 + fw, y1 + fh, 10, fill="#141922", outline="#3a4354", width=2)
        self.canvas.create_oval(x1 + 16, y1 + 14, x1 + 30, y1 + 28, fill=ACCENT, outline="")
        self.canvas.create_line(x1 + 8, y1 + fh - 12, x1 + 34, y1 + 38, x1 + 52, y1 + 56, x1 + 64, y1 + 44, x1 + fw - 8, y1 + fh - 12,
                                fill=MINT, width=3, smooth=True)
        self.canvas.create_text(cx, cy - 10, text="把照片拖到这里", fill=TEXT, font=self.app.fonts.h1)
        self.canvas.create_text(
            cx,
            cy + 24,
            text="或点击顶部「添加文件夹」开始  ·  支持 JPG / PNG / WEBP / BMP / TIFF / HEIC",
            fill=MUTED,
            font=self.app.fonts.small,
        )
        if DND_AVAILABLE:
            self.canvas.create_text(cx, cy + 48, text="直接拖入文件夹也可以，会自动扫描子目录", fill=FAINT, font=self.app.fonts.tiny)
        bx, by, bw, bh = cx - 78, cy + 72, 156, 36
        self._empty_button = (bx, by, bx + bw, by + bh)
        self._empty_action = "add"
        round_rect(self.canvas, bx, by, bx + bw, by + bh, 10, fill=ACCENT, outline="")
        self.canvas.create_text(cx, by + bh / 2, text="选择文件夹", fill="#1a1309", font=self.app.fonts.body_bold)
        self.canvas.create_text(
            cx,
            min(h - 28, by + bh + 28),
            text="照片只在本机分析，不会上传 · 可随时关闭",
            fill=FAINT,
            font=self.app.fonts.tiny,
        )
        if self._drop_active:
            self.canvas.create_rectangle(12, 12, w - 12, h - 12, outline=ACCENT, width=2, dash=(7, 5))
            self.canvas.create_text(cx, 32, text="松开鼠标，添加照片", fill=ACCENT, font=self.app.fonts.body_bold)

    # ------------------------------------------------------------------ events
    def _hit(self, x: float, y: float):
        y = self.canvas.canvasy(y)
        for x1, y1, x2, y2, key in reversed(self._hitboxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return key
        return None

    def _on_click(self, event):
        if self._empty_button:
            x1, y1, x2, y2 = self._empty_button
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                if self._empty_action == "filter":
                    self.app.set_filter("all")
                else:
                    self.app.add_folder()
                return
        key = self._hit(event.x, event.y)
        if key:
            self.app.select(key, additive=bool(event.state & 0x0004), range_select=bool(event.state & 0x0001))
            self.canvas.focus_set()

    def _on_double(self, event):
        key = self._hit(event.x, event.y)
        if key:
            self.app.open_preview(key)

    def _on_right(self, event):
        key = self._hit(event.x, event.y)
        if key:
            self.app.select(key)
            self.app.show_context_menu(event.x_root, event.y_root)
        return "break"

    def _on_motion(self, event):
        key = self._hit(event.x, event.y)
        if key != self._hover_id:
            self._hover_id = key
            self.schedule_render()
        self.canvas.configure(cursor="hand2" if key else "arrow")
        if key:
            photo = self.app.by_key.get(key)
            if photo:
                self.app.set_hover_path(photo)

    def _on_leave(self, _event):
        if self._hover_id:
            self._hover_id = None
            self.schedule_render()
        self.canvas.configure(cursor="arrow")
        self.app.set_hover_path(None)

    def set_drop_active(self, active: bool) -> None:
        if self._drop_active != active:
            self._drop_active = active
            self.schedule_render()

    def scroll_to_key(self, key: str) -> None:
        keys = [p.key for p in self.app.visible_photos]
        if key not in keys:
            return
        idx = keys.index(key)
        row = idx // max(1, self._cols)
        total = len(keys)
        rows = (total + self._cols - 1) // self._cols
        if rows <= 1:
            return
        target = max(0.0, min(1.0, row / (rows - 1)))
        visible = self.canvas.yview()
        if not (visible[0] <= target <= visible[1] - 0.05):
            self.canvas.yview_moveto(target)
            self.schedule_render()


# ---------------------------------------------------------------------------
# inspector
# ---------------------------------------------------------------------------
class Inspector(tk.Frame):
    PREVIEW_H = 230
    LOUPE = 156

    def __init__(self, master, app: "FocusSortApp"):
        super().__init__(master, bg=PANEL, width=382)
        self.pack_propagate(False)
        self.app = app
        self.photo: Photo | None = None
        self.full_image: Image.Image | None = None
        self.preview_image: Image.Image | None = None
        self.preview_tk: ImageTk.PhotoImage | None = None
        self.display_rect = (0, 0, 1, 1)
        self.focus = (0.5, 0.5)
        self.estimated_focus = (0.5, 0.5)
        self.heatmap = bool(app.settings.get("preview_heatmap", False))
        self.show_faces = bool(app.settings.get("preview_faces", True))
        self.show_focus = bool(app.settings.get("preview_focus", False))
        self._heat_img: Image.Image | None = None
        self._heat_key: tuple[str, int, int] | None = None
        self._face_rows: list[tk.Frame] = []
        self._face_photos: list[ImageTk.PhotoImage] = []
        self._loading = False
        self._build()

    def _build(self):
        pad = 16
        head = tk.Frame(self, bg=PANEL)
        head.pack(fill="x", padx=pad, pady=(14, 8))
        tk.Label(head, text="预览", bg=PANEL, fg=TEXT, font=self.app.fonts.h2).pack(side="left")
        self.nav_counter = tk.Label(head, text="", bg=PANEL, fg=FAINT, font=self.app.fonts.tiny)
        self.nav_counter.pack(side="right", padx=(0, 4))
        IconButton(head, "⛶", lambda: self.app.open_preview(), size=26, tooltip="大图预览 (空格)").pack(side="right", padx=(0, 2))
        IconButton(head, "›", lambda: self.app.navigate(1), size=26, tooltip="下一张 (→)").pack(side="right")
        IconButton(head, "‹", lambda: self.app.navigate(-1), size=26, tooltip="上一张 (←)").pack(side="right")
        self.hint_label = tk.Label(head, text="选择一张照片", bg=PANEL, fg=FAINT, font=self.app.fonts.tiny)
        self.hint_label.pack(side="right", pady=(4, 0))

        self.preview = tk.Canvas(
            self,
            bg="#090b10",
            height=self.PREVIEW_H,
            highlightthickness=1,
            highlightbackground=BORDER,
            bd=0,
            cursor="crosshair",
        )
        self.preview.pack(fill="x", padx=pad)
        self.preview.bind("<Configure>", lambda _e: self._redraw_preview())
        self.preview.bind("<Button-1>", self._on_preview_click)
        self.preview.bind("<Double-Button-1>", lambda _e: self.app.open_preview())

        bar = tk.Frame(self, bg=PANEL)
        bar.pack(fill="x", padx=pad, pady=(9, 0))
        self.seg = Segmented(
            bar,
            [("base", "原图"), ("heat", "细节热力")],
            "heat" if self.heatmap else "base",
            self._set_heat,
            height=30,
        )
        self.seg.pack(side="left")
        self.face_toggle = PillButton(
            bar,
            "人脸框",
            self._toggle_faces,
            variant="subtle" if not self.show_faces else "ghost",
            width=72,
            height=30,
            font=self.app.fonts.small_bold,
            radius=8,
        )
        self.face_toggle.pack(side="left", padx=(8, 0))
        self.focus_toggle = PillButton(
            bar,
            "焦点：开" if self.show_focus else "焦点：关",
            self._toggle_focus,
            variant="ghost" if self.show_focus else "subtle",
            width=74,
            height=30,
            font=self.app.fonts.small_bold,
            radius=8,
            tooltip="显示估计焦点小圆环（默认关闭；并非相机 AF 数据）",
        )
        self.focus_toggle.pack(side="left", padx=(8, 0))
        self.zoom_hint = tk.Label(self, text="点击预览定位 100% 细节", bg=PANEL, fg=FAINT, font=self.app.fonts.tiny)
        self.zoom_hint.pack(anchor="w", padx=pad, pady=(2, 0))

        # summary card
        self.summary = tk.Frame(self, bg=PANEL_2, highlightthickness=1, highlightbackground=BORDER)
        self.summary.pack(fill="x", padx=pad, pady=(10, 0))
        self.summary_inner = tk.Frame(self.summary, bg=PANEL_2)
        self.summary_inner.pack(fill="x", padx=14, pady=11)
        self.score_label = tk.Label(self.summary_inner, text="—", bg=PANEL_2, fg=TEXT, font=self.app.fonts.number)
        self.score_label.pack(side="left")
        right = tk.Frame(self.summary_inner, bg=PANEL_2)
        right.pack(side="left", padx=(12, 0), fill="x", expand=True)
        self.status_label = tk.Label(right, text="等待选择", bg=PANEL_2, fg=MUTED, font=self.app.fonts.h3, anchor="w")
        self.status_label.pack(fill="x")
        self.sub_label = tk.Label(right, text="清晰度分数 0–100", bg=PANEL_2, fg=FAINT, font=self.app.fonts.tiny, anchor="w")
        self.sub_label.pack(fill="x", pady=(3, 0))
        self.chips = tk.Frame(self.summary, bg=PANEL_2)
        self.chips.pack(fill="x", padx=14, pady=(0, 10))
        self.chip_labels: list[tk.Label] = []
        for _ in range(3):
            lab = tk.Label(self.chips, text="", bg="#1c2230", fg=MUTED, font=self.app.fonts.tiny, padx=8, pady=3)
            self.chip_labels.append(lab)
        self.dup_banner = tk.Label(
            self.summary,
            text="",
            bg="#241d36",
            fg=VIOLET,
            font=self.app.fonts.tiny,
            padx=10,
            pady=5,
            anchor="w",
            cursor="hand2",
        )
        self.dup_banner.bind("<Button-1>", lambda _e: self.app.show_duplicate_dialog())

        # actionable unreadable-file panel; hidden unless the selected photo
        # is in the error state.
        self.error_box = tk.Frame(self, bg="#1d1418", highlightthickness=1, highlightbackground="#43232d")
        self.error_title = tk.Label(
            self.error_box, text="无法读取", bg="#1d1418", fg=CORAL, font=self.app.fonts.body_bold, anchor="w"
        )
        self.error_title.pack(fill="x", padx=12, pady=(9, 1))
        self.error_detail = tk.Label(
            self.error_box, text="", bg="#1d1418", fg="#e0a4ad", font=self.app.fonts.tiny,
            anchor="w", justify="left", wraplength=330,
        )
        self.error_detail.pack(fill="x", padx=12)
        self.error_hint = tk.Label(
            self.error_box, text="", bg="#1d1418", fg=MUTED, font=self.app.fonts.tiny,
            anchor="w", justify="left", wraplength=330,
        )
        self.error_hint.pack(fill="x", padx=12, pady=(3, 7))
        self.error_actions = tk.Frame(self.error_box, bg="#1d1418")
        self.error_actions.pack(fill="x", padx=9, pady=(0, 9))
        PillButton(
            self.error_actions, "重试", self._retry_current, variant="ghost", width=54, height=28,
            font=self.app.fonts.small,
        ).pack(side="left", padx=2)
        PillButton(
            self.error_actions, "从列表移除", self._remove_current, variant="subtle", width=88, height=28,
            font=self.app.fonts.small,
        ).pack(side="left", padx=2)
        PillButton(
            self.error_actions, "打开文件夹", self._reveal_current, variant="subtle", width=88, height=28,
            font=self.app.fonts.small,
        ).pack(side="left", padx=2)

        # face section
        face_head = tk.Frame(self, bg=PANEL)
        face_head.pack(fill="x", padx=pad, pady=(12, 4))
        self.face_head_title = tk.Label(face_head, text="面部", bg=PANEL, fg=TEXT_SOFT, font=self.app.fonts.small_bold)
        self.face_head_title.pack(side="left")
        self.face_count_label = tk.Label(face_head, text="未分析", bg=PANEL, fg=FAINT, font=self.app.fonts.tiny)
        self.face_count_label.pack(side="right")
        self.face_box = tk.Frame(self, bg=PANEL)
        self.face_box.pack(fill="x", padx=pad)
        self.face_empty = tk.Label(
            self.face_box,
            text="选择照片后显示人脸清晰度与遮挡估计",
            bg=PANEL,
            fg=FAINT,
            font=self.app.fonts.tiny,
            anchor="w",
            justify="left",
        )
        self.face_empty.pack(fill="x", pady=(2, 0))

        # loupe
        loupe_head = tk.Frame(self, bg=PANEL)
        loupe_head.pack(fill="x", padx=pad, pady=(12, 4))
        tk.Label(loupe_head, text="100% 细节", bg=PANEL, fg=TEXT_SOFT, font=self.app.fonts.small_bold).pack(side="left")
        self.loupe_meta = tk.Label(loupe_head, text="", bg=PANEL, fg=FAINT, font=self.app.fonts.tiny)
        self.loupe_meta.pack(side="right")
        self.loupe = tk.Canvas(self, width=self.LOUPE, height=self.LOUPE, bg="#090b10", highlightthickness=1, highlightbackground=BORDER, bd=0)
        self.loupe.pack(padx=pad, pady=(0, 6))

        self.note = tk.Label(
            self,
            text="提示：焦点标记默认关闭；点「焦点」才显示细小圆环。它由单张图的局部清晰区域/人脸推断，"
                 "不是相机 AF 焦点；100% 细节最可靠。",
            bg=PANEL,
            fg=FAINT,
            font=self.app.fonts.tiny,
            wraplength=340,
            justify="left",
        )
        self.note.pack(fill="x", padx=pad, pady=(2, 0))

    # ------------------------------------------------------------------ public
    def show_photo(self, photo: Photo) -> None:
        self.photo = photo
        try:
            keys = [p.key for p in self.app.visible_photos]
            self.nav_counter.configure(text=f"{keys.index(photo.key) + 1} / {len(keys)}")
        except ValueError:
            self.nav_counter.configure(text="")
        self.full_image = None
        self.preview_image = None
        self.preview_tk = None
        self._heat_img = None
        self._heat_key = None
        self._face_photos.clear()
        self._clear_face_rows()
        self.focus = (0.5, 0.5)
        self.estimated_focus = (0.5, 0.5)
        if photo.result and photo.result.get("focus"):
            f = photo.result["focus"]
            self.estimated_focus = (float(f.get("x", 0.5)), float(f.get("y", 0.5)))
            self.focus = self.estimated_focus
        elif photo.result and photo.result.get("faces"):
            faces = photo.result["faces"]
            main = max(faces, key=lambda f: f.get("width", 0))
            if not main.get("small"):
                bx, by, bw, bh = main["box"]
                self.focus = (bx + bw / 2, by + bh / 2)
                self.estimated_focus = self.focus
        self._set_loading(not photo.is_error)
        if photo.is_error:
            self.hint_label.configure(text="文件不可读")
        self._update_summary()
        self._update_error_box()
        self._update_face_section()
        self._redraw_preview()
        self._redraw_loupe()

    def set_image(self, full: Image.Image, preview: Image.Image) -> None:
        self._loading = False
        self.full_image = full
        self.preview_image = preview
        self.preview_tk = None
        self._redraw_preview()
        self._redraw_loupe()
        self._update_summary()

    def update_result(self) -> None:
        if not self.photo:
            return
        self._update_summary()
        self._update_error_box()
        self._update_face_section()
        self._redraw_preview()
        self._redraw_loupe()

    def clear(self) -> None:
        self.photo = None
        self.full_image = None
        self.preview_image = None
        self.preview_tk = None
        self._heat_img = None
        self.focus = (0.5, 0.5)
        self._clear_face_rows()
        self.error_box.pack_forget()
        try:
            self.face_head_title.configure(text="面部")
        except tk.TclError:
            pass
        self.face_count_label.configure(text="未分析")
        self.face_empty.pack(fill="x", pady=(2, 0))
        self.score_label.configure(text="—", fg=TEXT)
        self.status_label.configure(text="等待选择", fg=MUTED)
        self.sub_label.configure(text="清晰度分数 0–100")
        for lab in self.chip_labels:
            lab.pack_forget()
        self._redraw_preview()
        self._redraw_loupe()

    # ------------------------------------------------------------------ summary
    def _set_loading(self, loading: bool) -> None:
        self._loading = loading
        self.hint_label.configure(text="载入原图…" if loading else "点击预览定位细节")

    def _update_summary(self) -> None:
        p = self.photo
        if not p:
            return
        if p.is_error:
            self.hint_label.configure(text="文件不可读")
        else:
            self.hint_label.configure(text="载入原图…" if self._loading else "点击预览定位细节")
        if not p.result:
            self.score_label.configure(text="—", fg=TEXT)
            self.status_label.configure(text="分析中…" if p.state != "error" else "无法读取", fg=MUTED if p.state != "error" else CORAL)
            self.sub_label.configure(text="正在计算清晰度与人脸")
        elif p.result.get("error"):
            self.score_label.configure(text="!", fg=CORAL)
            self.status_label.configure(text="无法读取图片", fg=CORAL)
            self.sub_label.configure(text=ellipsize(str(p.result.get("error", "")), self.app.font_obj(self.app.fonts.tiny), 330))
        else:
            status = p.status(self.app.threshold)
            score = float(p.result.get("score", 0))
            color = STATUS_COLORS.get(status, MUTED)
            self.score_label.configure(text=f"{score:.0f}", fg=color)
            self.status_label.configure(text=STATUS_LABELS.get(status, ""), fg=color)
            bits = [f"全局 {p.result.get('global_score', 0):.0f}"]
            if p.result.get("face_score") is not None:
                bits.append(f"人脸 {p.result['face_score']:.0f}")
            focus = p.result.get("focus") or {}
            if focus:
                bits.append("焦点·" + ("人脸" if focus.get("source") == "face" else "细节"))
            if p.result.get("low_detail"):
                bits.append("细节较少")
            if p.result.get("confidence", 1) < 0.35:
                bits.append("参考性低")
            self.sub_label.configure(text="  ·  ".join(bits))
        self._update_chips()
        self._update_error_box()

    def _update_error_box(self) -> None:
        p = self.photo
        if not p or not p.is_error:
            self.error_box.pack_forget()
            try:
                self.face_head_title.configure(text="面部")
            except tk.TclError:
                pass
            return
        reason = p.error_reason()
        hint = p.error_hint()
        code = (p.result or {}).get("error_code", "")
        self.error_title.configure(text=f"无法读取  ·  {reason}")
        detail = code
        technical = str((p.result or {}).get("error_detail", "")).strip()
        if technical:
            detail = f"{code}  ·  {technical}" if code else technical
        self.error_detail.configure(text=ellipsize(detail, self.app.font_obj(self.app.fonts.tiny), 330))
        self.error_hint.configure(text=hint)
        self.face_head_title.configure(text="错误详情")
        self.face_count_label.configure(text="")
        # Keep the panel directly under the compact error summary.
        try:
            self.error_box.pack(fill="x", padx=16, pady=(8, 0), before=self.face_empty.master)
        except tk.TclError:
            self.error_box.pack(fill="x", padx=16, pady=(8, 0))

    def _retry_current(self) -> None:
        if self.photo:
            self.app.retry_photo(self.photo.key)

    def _remove_current(self) -> None:
        if self.photo:
            self.app.remove_photo(self.photo.key)

    def _reveal_current(self) -> None:
        if self.photo:
            self.app.reveal(self.photo)

    def _update_chips(self) -> None:
        for lab in self.chip_labels:
            lab.pack_forget()
        self.dup_banner.pack_forget()
        p = self.photo
        if not p or not p.result or p.result.get("error"):
            return
        r = p.result
        values = []
        if r.get("face_count"):
            values.append((f"{r['face_count']} 人", BLUE))
        if r.get("occluded"):
            values.append(("有遮挡", BLUE))
        if r.get("low_detail"):
            values.append(("细节少", AMBER))
        if r.get("confidence", 1) < 0.35:
            values.append(("参考性低", MUTED))
        for lab, (text, color) in zip(self.chip_labels, values[:3]):
            lab.configure(text=text, fg=color, bg="#1b2333" if color != MUTED else "#1a1e27")
            lab.pack(side="left", padx=(0, 6))
        group = self.app.duplicate_group_for(p.key) if p else []
        if group:
            self.dup_banner.configure(text=f"同一照片有 {len(group)} 个格式版本，点击管理去重")
            self.dup_banner.pack(fill="x", padx=14, pady=(0, 10))
        else:
            self.dup_banner.pack_forget()

    # ------------------------------------------------------------------ faces
    def _clear_face_rows(self) -> None:
        for row in self._face_rows:
            try:
                row.destroy()
            except tk.TclError:
                pass
        self._face_rows = []

    def _update_face_section(self) -> None:
        self._clear_face_rows()
        p = self.photo
        if not p or not p.result or p.result.get("error"):
            self.face_count_label.configure(text="未分析" if not p else ("" if p.is_error else "—"))
            if p and p.is_error:
                self.face_empty.pack_forget()
            else:
                self.face_empty.pack(fill="x", pady=(2, 0))
                self.face_empty.configure(text="选择照片后显示人脸清晰度与遮挡估计")
            return
        faces = p.result.get("faces", [])
        if not faces:
            self.face_empty.configure(text="未检测到人脸，按整体清晰度判断")
            self.face_empty.pack(fill="x", pady=(2, 0))
            self.face_count_label.configure(text="0 张脸")
            return
        self.face_empty.pack_forget()
        self.face_count_label.configure(text=f"{len(faces)} 张脸")
        show = faces[:3]
        src = self.full_image or self.preview_image
        for idx, face in enumerate(show):
            row = self._make_face_row(p, face, idx, src)
            row.pack(fill="x", pady=(0, 4))
            self._face_rows.append(row)
        if len(faces) > len(show):
            more = tk.Label(self.face_box, text=f"另有 {len(faces) - len(show)} 张人脸…", bg=PANEL, fg=FAINT, font=self.app.fonts.tiny, anchor="w")
            more.pack(fill="x")
            self._face_rows.append(more)  # type: ignore[arg-type]

    def _make_face_row(self, p: Photo, face: dict, idx: int, src: Image.Image | None) -> tk.Frame:
        row = tk.Frame(self.face_box, bg="#171b24", cursor="hand2", highlightthickness=1, highlightbackground="#171b24")
        row.grid_columnconfigure(1, weight=1)
        thumb = self._face_thumb(p, face, src)
        if thumb is not None:
            lab = tk.Label(row, image=thumb, bg="#171b24", bd=0, padx=0, pady=0)
            lab.image = thumb
            lab.grid(row=0, column=0, rowspan=2, padx=(7, 9), pady=6, sticky="n")
        score = face.get("score", 0)
        face_status = "清晰" if score >= self.app.threshold else ("待复核" if score >= self.app.threshold - 15 else "模糊")
        color = MINT if face_status == "清晰" else (AMBER if face_status == "待复核" else CORAL)
        top = tk.Frame(row, bg="#171b24")
        top.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(5, 0))
        tk.Label(top, text=f"面部 {idx + 1}", bg="#171b24", fg=TEXT_SOFT, font=self.app.fonts.small_bold).pack(side="left")
        tk.Label(top, text=f"{score:.0f} 分", bg="#171b24", fg=color, font=self.app.fonts.small_bold).pack(side="left", padx=(7, 0))
        tk.Label(top, text=face_status, bg="#171b24", fg=color, font=self.app.fonts.tiny).pack(side="right", padx=(0, 6))
        occ = face.get("occlusion", {})
        occ_color = BLUE if occ.get("occluded") else MUTED
        if face.get("small"):
            occ_text = "人脸较小 · 遮挡仅供参考"
        else:
            occ_text = occ.get("label", "未见明显遮挡")
            if occ.get("occluded"):
                occ_text = f"{occ_text}  {occ.get('score', 0) * 100:.0f}%"
            if face.get("eyes") is not None:
                occ_text += f"  ·  眼部 {int(face.get('eyes', 0))}/2"
            reason = str(occ.get("reason") or "").strip()
            if occ.get("occluded") and reason:
                occ_text += f"  ·  {reason}"
        full_occ_text = occ_text
        occ_text = ellipsize(occ_text, self.app.font_obj(self.app.fonts.tiny), 285)
        occ_label = tk.Label(row, text=occ_text, bg="#171b24", fg=occ_color, font=self.app.fonts.tiny, anchor="w")
        occ_label.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(1, 5), padx=(0, 6))
        if full_occ_text != occ_text:
            ToolTip(occ_label, full_occ_text)

        def click(_e=None, f=face):
            bx, by, bw, bh = f["box"]
            self.set_focus((bx + bw / 2, by + bh / 2))

        def bind_tree(widget):
            widget.bind("<Button-1>", click)
            for child in widget.winfo_children():
                bind_tree(child)

        bind_tree(row)
        row.bind("<Enter>", lambda _e: row.configure(highlightbackground=BORDER_LIGHT))
        row.bind("<Leave>", lambda _e: row.configure(highlightbackground="#171b24"))
        return row

    def _face_thumb(self, p: Photo, face: dict, src: Image.Image | None) -> ImageTk.PhotoImage | None:
        if src is None:
            return None
        try:
            w, h = src.size
            bx, by, bw, bh = face["box"]
            box = (int(bx * w), int(by * h), int((bx + bw) * w), int((by + bh) * h))
            crop = src.crop(box).convert("RGB")
            crop = ImageOps.fit(crop, (42, 42), method=Image.Resampling.LANCZOS, centering=(0.5, 0.42))
            crop = rounded_image(crop, 8)
            tkimg = ImageTk.PhotoImage(crop)
            self._face_photos.append(tkimg)
            return tkimg
        except Exception:
            return None

    # ------------------------------------------------------------------ preview
    def _set_heat(self, value: str) -> None:
        self.heatmap = value == "heat"
        self.app.save_preview_prefs()
        self._redraw_preview()

    def _toggle_faces(self) -> None:
        self.show_faces = not self.show_faces
        # Rebuild the button so its variant changes cleanly.
        self.face_toggle.destroy()
        self.face_toggle = PillButton(
            self.seg.master,
            "人脸框",
            self._toggle_faces,
            variant="ghost" if self.show_faces else "subtle",
            width=72,
            height=30,
            font=self.app.fonts.small_bold,
            radius=8,
        )
        self.face_toggle.pack(side="left", padx=(8, 0), before=self.focus_toggle)
        self.app.save_preview_prefs()
        self._redraw_preview()

    def _toggle_focus(self) -> None:
        self.show_focus = not self.show_focus
        self.focus_toggle.destroy()
        self.focus_toggle = PillButton(
            self.seg.master,
            "焦点：开" if self.show_focus else "焦点：关",
            self._toggle_focus,
            variant="ghost" if self.show_focus else "subtle",
            width=74,
            height=30,
            font=self.app.fonts.small_bold,
            radius=8,
            tooltip="显示估计焦点小圆环（默认关闭；并非相机 AF 数据）",
        )
        self.focus_toggle.pack(side="left", padx=(8, 0))
        self.app.save_preview_prefs()
        self._redraw_preview()

    def _content_rect(self, img_w: int, img_h: int, cw: int, ch: int) -> tuple[int, int, int, int]:
        scale = min(cw / img_w, ch / img_h)
        dw, dh = max(1, int(img_w * scale)), max(1, int(img_h * scale))
        return (cw - dw) // 2, (ch - dh) // 2, dw, dh

    def _redraw_preview(self) -> None:
        c = self.preview
        c.delete("all")
        cw, ch = max(1, c.winfo_width()), max(1, c.winfo_height())
        if not self.photo:
            c.create_text(cw / 2, ch / 2 - 10, text="选择一张照片", fill=MUTED, font=self.app.fonts.body)
            c.create_text(cw / 2, ch / 2 + 16, text="查看 100% 细节、热力图与人脸框", fill=FAINT, font=self.app.fonts.tiny)
            self.display_rect = (0, 0, cw, ch)
            return
        if self.photo.is_error:
            c.create_text(cw / 2, ch / 2 - 18, text="无法读取", fill=CORAL, font=self.app.fonts.h2)
            reason = ellipsize(self.photo.error_reason(), self.app.font_obj(self.app.fonts.small), cw - 24)
            c.create_text(cw / 2, ch / 2 + 10, text=reason, fill="#d9a2ab", font=self.app.fonts.small, width=max(80, cw - 28))
            c.create_text(cw / 2, ch / 2 + 34, text="可直接重试、移除或打开所在文件夹", fill=FAINT, font=self.app.fonts.tiny)
            self.display_rect = (0, 0, cw, ch)
            return
        if self.preview_image is None:
            thumb = self.app.get_card_thumb(self.photo, 320, 220)
            if thumb is not None:
                self._placeholder_tk = thumb
                c.create_image(cw / 2, ch / 2, image=thumb, anchor="center")
            else:
                c.create_text(cw / 2, ch / 2, text="载入中…", fill=MUTED, font=self.app.fonts.body)
            self.display_rect = (0, 0, cw, ch)
            return

        img = self.preview_image
        dx, dy, dw, dh = self._content_rect(img.width, img.height, cw, ch)
        self.display_rect = (dx, dy, dw, dh)
        display = img
        if self.heatmap:
            display = self._heatmap_image(img, (dw, dh))
        display = display.resize((dw, dh), Image.Resampling.LANCZOS)
        self.preview_tk = ImageTk.PhotoImage(display)
        c.create_image(dx, dy, image=self.preview_tk, anchor="nw")
        # subtle border
        c.create_rectangle(dx, dy, dx + dw - 1, dy + dh - 1, outline="#2a3140", width=1)

        if self.show_faces and self.photo.result:
            for idx, face in enumerate(self.photo.result.get("faces", [])):
                bx, by, bw, bh = face["box"]
                x1 = dx + bx * dw
                y1 = dy + by * dh
                x2 = dx + (bx + bw) * dw
                y2 = dy + (by + bh) * dh
                if face.get("occlusion", {}).get("occluded"):
                    color = BLUE
                elif face.get("score", 100) < self.app.threshold - 10:
                    color = CORAL
                else:
                    color = MINT
                c.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
                tag = str(idx + 1)
                tw = 20
                c.create_rectangle(x1, y1 - 20, x1 + tw, y1, fill=color, outline="")
                c.create_text(x1 + tw / 2, y1 - 10, text=tag, fill="#0c1017", font=self.app.fonts.tiny)

        if self.heatmap:
            c.create_text(dx + 8, dy + dh - 10, text="细节少", fill="#ff8b98", font=self.app.fonts.tiny, anchor="sw")
            c.create_text(dx + dw - 8, dy + dh - 10, text="细节多", fill="#8ef0bd", font=self.app.fonts.tiny, anchor="se")

        # Optional estimated focus.  Off by default; when enabled it is only a
        # small thin ring plus a tiny label, never a large crosshair.
        if self.show_focus and self.photo.result:
            focus = self.photo.result.get("focus") or {}
            if focus:
                ex = dx + float(focus.get("x", self.estimated_focus[0])) * dw
                ey = dy + float(focus.get("y", self.estimated_focus[1])) * dh
                c.create_oval(ex - 8, ey - 8, ex + 8, ey + 8, outline="#d9bd87", width=1, dash=(2, 4))
                c.create_oval(ex - 1.5, ey - 1.5, ex + 1.5, ey + 1.5, fill="#d9bd87", outline="")
                label = "人脸" if focus.get("source") == "face" else "细节"
                if focus.get("label") and focus.get("source") == "face":
                    label = str(focus.get("label"))
                text = f"估计焦点 · {label}"
                tx, ty, anchor = ex + 11, ey, "w"
                if focus.get("source") == "face" and self.photo.result:
                    try:
                        idx = int(str(focus.get("label", "")).split()[-1]) - 1
                        face = self.photo.result.get("faces", [])[idx]
                        bx, by, bw, bh = face["box"]
                        tx = dx + (bx + bw / 2) * dw
                        ty = dy + (by + bh) * dh + 3
                        anchor = "n"
                        if ty > dy + dh - 14:
                            ty = dy + by * dh - 3
                            anchor = "s"
                    except (ValueError, IndexError, KeyError, TypeError):
                        pass
                if anchor == "w" and tx + self.app.measure(text, self.app.fonts.tiny) > dx + dw - 4:
                    tx, anchor = ex - 11, "e"
                c.create_text(tx, ty, text=text, fill="#c9ad78", font=self.app.fonts.tiny, anchor=anchor)

        # The loupe target is also hidden by default; clicking still moves the
        # 100% loupe without adding permanent clutter.
        if self.show_focus:
            fx = dx + self.focus[0] * dw
            fy = dy + self.focus[1] * dh
            c.create_oval(fx - 4, fy - 4, fx + 4, fy + 4, outline=WHITE, width=1)
            c.create_oval(fx - 1, fy - 1, fx + 1, fy + 1, fill=WHITE, outline="")

    def _heatmap_image(self, img: Image.Image, size: tuple[int, int]) -> Image.Image:
        key = (self.photo.key if self.photo else "", size[0], size[1])
        if self._heat_img is not None and self._heat_key == key:
            return self._heat_img
        try:
            base = img.resize(size, Image.Resampling.LANCZOS).convert("RGB")
            arr = np.asarray(base)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            k = 25
            mean = cv2.boxFilter(lap, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
            mean_sq = cv2.boxFilter(lap * lap, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
            var = np.maximum(mean_sq - mean * mean, 0)
            var = cv2.GaussianBlur(np.log1p(var), (0, 0), 5)
            lo, hi = np.percentile(var, 4), np.percentile(var, 96)
            norm = np.clip((var - lo) / max(1e-6, hi - lo), 0, 1)
            # red -> amber -> green
            stops = np.array([0.0, 0.35, 0.68, 1.0])
            colors = np.array([[255, 77, 94], [246, 196, 91], [176, 232, 128], [56, 226, 156]], np.float32)
            lut = np.stack([np.interp(np.linspace(0, 1, 256), stops, colors[:, i]) for i in range(3)], axis=1).astype(np.uint8)
            idx = (norm * 255).astype(np.uint8)
            heat = lut[idx]
            out = (arr.astype(np.float32) * 0.70 + heat.astype(np.float32) * 0.30).astype(np.uint8)
            self._heat_img = Image.fromarray(out)
            self._heat_key = key
            return self._heat_img
        except Exception:
            return img

    def _on_preview_click(self, event):
        if not self.photo or not self.preview_image:
            return
        dx, dy, dw, dh = self.display_rect
        if not (dx <= event.x <= dx + dw and dy <= event.y <= dy + dh):
            return
        nx = (event.x - dx) / max(1, dw)
        ny = (event.y - dy) / max(1, dh)
        self.set_focus((max(0, min(1, nx)), max(0, min(1, ny))))

    def set_focus(self, focus: tuple[float, float]) -> None:
        self.focus = (max(0.0, min(1.0, focus[0])), max(0.0, min(1.0, focus[1])))
        self._redraw_preview()
        self._redraw_loupe()

    # ------------------------------------------------------------------ loupe
    def _redraw_loupe(self) -> None:
        c = self.loupe
        c.delete("all")
        size = self.LOUPE
        c.create_text(size / 2, size / 2, text="选择照片后\n查看原图像素", fill=FAINT, font=self.app.fonts.tiny, justify="center")
        if self.photo and self.photo.is_error:
            c.delete("all")
            c.create_text(size / 2, size / 2, text="无法读取\n请重试或移除", fill=CORAL, font=self.app.fonts.tiny, justify="center")
            self.loupe_meta.configure(text="")
            return
        src = self.full_image or self.preview_image
        if not self.photo or src is None:
            return
        try:
            w, h = src.size
            cx, cy = self.focus[0] * w, self.focus[1] * h
            half = size / 2
            left = max(0, min(w - size, cx - half))
            top = max(0, min(h - size, cy - half))
            if w < size or h < size:
                crop = src.crop((0, 0, w, h)).resize((size, size), Image.Resampling.NEAREST)
            else:
                crop = src.crop((int(left), int(top), int(left) + size, int(top) + size)).convert("RGB")
            self._loupe_tk = ImageTk.PhotoImage(crop)
            c.create_image(0, 0, image=self._loupe_tk, anchor="nw")
            c.create_line(size / 2 - 10, size / 2, size / 2 + 10, size / 2, fill="#ffffff", width=1)
            c.create_line(size / 2, size / 2 - 10, size / 2, size / 2 + 10, fill="#ffffff", width=1)
            c.create_text(8, 8, text="100%", fill=WHITE, font=self.app.fonts.tiny, anchor="nw")
            coord = f"x {int(cx)}  y {int(cy)}"
            c.create_text(size - 8, size - 8, text=coord, fill="#d7dbe5", font=self.app.fonts.tiny, anchor="se")
            self.loupe_meta.configure(text=f"{w}×{h} 原图")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# main application
# ---------------------------------------------------------------------------
class FocusSortApp:
    def __init__(self):
        enable_dpi_awareness()
        ensure_data_dir()
        self.settings = load_settings()
        # Visual-first default: every fresh launch shows the photo unobscured,
        # regardless of overlay toggles left on in a previous session.
        for overlay_key in ("preview_heatmap", "preview_faces", "preview_focus"):
            self.settings[overlay_key] = False
        self.threshold = float(self.settings.get("threshold", DEFAULT_THRESHOLD))
        self.sort_mode = str(self.settings.get("sort", "import"))
        self.filter_mode = str(self.settings.get("filter", "all"))
        self.photos: list[Photo] = []
        self.by_key: dict[str, Photo] = {}
        self.visible_photos: list[Photo] = []
        self.selected_key: str | None = None
        self.selected_keys: set[str] = set()
        self.selection_anchor: str | None = None
        self.dup_groups: list[list[Photo]] = []
        self.dup_of: dict[str, int] = {}
        self.dup_count = 0
        self.preview_window = None
        self.settings_window = None
        self._hover_path: str | None = None
        self._closing = False
        self._analysis_jobs: dict[Future, str] = {}
        self._jobs_lock = threading.Lock()
        # Worker threads never call Tk.  They push callbacks here and the main
        # thread drains the queue, which avoids a Tcl deadlock if the user
        # closes the window while an image is still loading.
        self._callback_queue: queue.Queue[tuple[Any, tuple]] = queue.Queue()
        self._callback_job = None
        self._queue_total = 0
        self._queue_done = 0
        self._active = False
        self._cancel = False
        self._sort_frozen = False
        self._refresh_job = None
        self._selection_job = None
        self._preview_token = 0
        self._last_dir = str(self.settings.get("last_dir") or (Path.home() / "Pictures"))

        cv2.setNumThreads(1)
        self.analyzer = Analyzer(CACHE_DIR)
        self.executor = ThreadPoolExecutor(max_workers=min(4, max(1, (os.cpu_count() or 4) - 1)), thread_name_prefix="focus-analyze")
        self.preview_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="focus-preview")

        if DND_AVAILABLE:
            try:
                self.root = TkinterDnD.Tk()
            except Exception:
                globals()["DND_AVAILABLE"] = False
                self.root = tk.Tk()
        else:
            self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.configure(bg=BG)
        self.root.minsize(1240, 840)
        self.root.geometry(self._safe_geometry())
        self.root.report_callback_exception = self._report_callback_exception
        enable_dark_titlebar(self.root)
        try:
            self.root.iconbitmap(default=str(APP_DIR / "assets" / "icon.ico"))
        except Exception:
            pass
        self.fonts = Fonts(self.root)
        self.tkfont = tkfont
        self._font_cache: dict[tuple, tkfont.Font] = {}
        self.toast = Toast(self.root)
        self._build()
        self._bind_keys()
        self._update_counts()
        self._refresh_visible()
        self._refresh_grid()
        self.inspector.clear()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(300, self._startup_note)
        self._callback_job = self.root.after(40, self._drain_callbacks)

    # ------------------------------------------------------------------ fonts
    def font_obj(self, font: tuple) -> tkfont.Font:
        f = self._font_cache.get(font)
        if f is None:
            f = tkfont.Font(root=self.root, font=font)
            self._font_cache[font] = f
        return f

    def measure(self, text: str, font: tuple) -> int:
        return self.font_obj(font).measure(text)

    # ------------------------------------------------------------------ geometry
    def _safe_geometry(self) -> str:
        geo = str(self.settings.get("geometry") or "1560x960")
        try:
            w, h = [int(v) for v in geo.lower().split("x")]
        except ValueError:
            w, h = 1560, 960
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w = min(w, sw - 40)
        h = min(h, sh - 80)
        w = max(1180, w)
        h = max(760, h)
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2 - 18)
        return f"{w}x{h}+{x}+{y}"

    # ------------------------------------------------------------------ build UI
    def _build(self) -> None:
        self._build_topbar()
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)
        self._build_sidebar(body)
        center = tk.Frame(body, bg=BG)
        center.pack(side="left", fill="both", expand=True)
        self.grid = PhotoGrid(center, self)
        self.grid.pack(fill="both", expand=True)
        self.batch_bar = tk.Frame(self.grid, bg="#202634", highlightthickness=1, highlightbackground=BORDER_LIGHT)
        self.batch_label = tk.Label(self.batch_bar, text="", bg="#202634", fg=TEXT, font=self.fonts.small)
        self.batch_label.pack(side="left", padx=(14, 10), pady=8)
        PillButton(self.batch_bar, "全选", self.select_all, variant="subtle", width=56, height=30, font=self.fonts.small).pack(side="left", padx=4)
        PillButton(self.batch_bar, "移入回收站", self.delete_selected, variant="danger", width=104, height=30, font=self.fonts.small).pack(side="left", padx=4)
        PillButton(self.batch_bar, "取消选择", self.clear_selection, variant="subtle", width=76, height=30, font=self.fonts.small).pack(side="left", padx=4)
        self.inspector = Inspector(body, self)
        self.inspector.pack(side="right", fill="y")
        self._build_statusbar()

    def _build_topbar(self) -> None:
        top = tk.Frame(self.root, bg=PANEL, height=66)
        top.pack(fill="x")
        top.pack_propagate(False)
        left = tk.Frame(top, bg=PANEL)
        left.pack(side="left", padx=(20, 0), pady=12)
        logo = tk.Canvas(left, width=36, height=36, bg=PANEL, highlightthickness=0, bd=0)
        logo.pack(side="left")
        logo.create_oval(3, 3, 33, 33, outline=ACCENT, width=2)
        logo.create_oval(12, 12, 24, 24, outline=ACCENT, width=2)
        logo.create_oval(16, 16, 20, 20, fill=ACCENT, outline="")
        logo.create_line(18, 3, 18, 8, fill=ACCENT, width=2)
        name = tk.Frame(left, bg=PANEL)
        name.pack(side="left", padx=(9, 0))
        tk.Label(name, text="焦点", bg=PANEL, fg=TEXT, font=self.fonts.h1).pack(anchor="w")
        tk.Label(name, text="FocusSort · 照片清晰度快筛", bg=PANEL, fg=FAINT, font=self.fonts.tiny).pack(anchor="w", pady=(1, 0))

        right = tk.Frame(top, bg=PANEL)
        right.pack(side="right", padx=(0, 16), pady=12)
        self.export_btn = PillButton(right, "导出报告", self.export_report, variant="ghost", width=94, height=36, icon="⇩")
        self.export_btn.pack(side="right", padx=(8, 0))
        self.preview_btn = PillButton(right, "大图预览", self.open_preview, variant="ghost", width=104, height=36, icon="⛶",
                                      tooltip="打开大图预览，连续翻阅 (空格)")
        self.preview_btn.pack(side="right", padx=(8, 0))
        self.settings_btn = IconButton(right, "⚙", self.show_settings, size=36, tooltip="设置、会话恢复与缓存清理", fg=MUTED)
        self.settings_btn.pack(side="right", padx=(4, 0))
        self.help_btn = IconButton(right, "?", self.show_help, size=36, tooltip="使用说明与快捷键  (?)", fg=MUTED)
        self.help_btn.pack(side="right", padx=(4, 0))
        self.folder_btn = PillButton(right, "文件夹", self.add_folder, variant="primary", width=106, height=36, icon="＋")
        self.folder_btn.pack(side="right", padx=(8, 0))
        self.files_btn = PillButton(right, "照片", self.add_files, variant="ghost", width=86, height=36, icon="＋")
        self.files_btn.pack(side="right")

        self.top_info = tk.Label(top, text="", bg=PANEL, fg=MUTED, font=self.fonts.small)
        self.top_info.pack(side="left", padx=(28, 0))
        self.stop_btn = PillButton(top, "停止", self.cancel_analysis, variant="subtle", width=54, height=30, font=self.fonts.small)
        self.stop_btn.pack_propagate(False)

        self.progress_line = tk.Canvas(self.root, height=3, bg=BG, highlightthickness=0, bd=0)
        self.progress_line.pack(fill="x")
        self.progress_line.bind("<Configure>", lambda _e: self._draw_progress())

    def _build_sidebar(self, parent: tk.Frame) -> None:
        side = tk.Frame(parent, bg=PANEL, width=212)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        pad = 16
        tk.Label(side, text="图库", bg=PANEL, fg=FAINT, font=self.fonts.tiny).pack(anchor="w", padx=pad, pady=(16, 6))
        self.filter_rows: dict[str, tuple[tk.Frame, tk.Label, tk.Label, tk.Frame]] = {}
        filters = [
            ("all", "全部照片", "◫"),
            ("sharp", "清晰", "●"),
            ("review", "待复核", "◐"),
            ("blur", "模糊", "○"),
            ("error", "异常 / 无法读取", "!"),
            ("occluded", "有遮挡", "◇"),
            ("duplicates", "同图重复", "⧉"),
        ]
        for key, label, glyph in filters:
            row = tk.Frame(side, bg=PANEL, cursor="hand2")
            row.pack(fill="x", padx=8, pady=1)
            bar = tk.Frame(row, bg=PANEL, width=3, height=28)
            bar.pack(side="left", fill="y", padx=(0, 8))
            gl = tk.Label(row, text=glyph, bg=PANEL, fg=MUTED, font=self.fonts.small, width=2)
            gl.pack(side="left")
            lab = tk.Label(row, text=label, bg=PANEL, fg=TEXT_SOFT, font=self.fonts.body, anchor="w")
            lab.pack(side="left", fill="x", expand=True)
            cnt = tk.Label(row, text="0", bg=PANEL, fg=FAINT, font=self.fonts.tiny)
            cnt.pack(side="right", padx=(0, 8))
            for wgt in (row, bar, gl, lab, cnt):
                wgt.bind("<Button-1>", lambda _e, k=key: self.set_filter(k))
                wgt.bind("<Enter>", lambda _e, r=row: r.configure(bg=CARD_HOVER))
                wgt.bind("<Leave>", lambda _e, r=row: r.configure(bg=PANEL))
            self.filter_rows[key] = (row, lab, cnt, bar)

        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=pad, pady=(14, 12))
        head = tk.Frame(side, bg=PANEL)
        head.pack(fill="x", padx=pad)
        tk.Label(head, text="清晰判定线", bg=PANEL, fg=TEXT_SOFT, font=self.fonts.small_bold).pack(side="left")
        self.threshold_label = tk.Label(head, text=f"{self.threshold:.0f}", bg=PANEL, fg=ACCENT, font=self.fonts.small_bold)
        self.threshold_label.pack(side="right")
        self.slider = ThresholdSlider(side, self.threshold, self._on_threshold, width=178)
        self.slider.pack(padx=pad, pady=(4, 2))
        presets = tk.Frame(side, bg=PANEL)
        presets.pack(fill="x", padx=pad)
        for label, val in (("宽松", 45), ("标准", 60), ("严格", 72)):
            b = tk.Label(presets, text=label, bg="#1a1f2a", fg=MUTED, font=self.fonts.tiny, padx=8, pady=4, cursor="hand2")
            b.pack(side="left", padx=(0, 5))
            b.bind("<Button-1>", lambda _e, v=val: self.set_threshold(v))
            b.bind("<Enter>", lambda _e, w=b: w.configure(bg="#252d3c", fg=TEXT))
            b.bind("<Leave>", lambda _e, w=b: w.configure(bg="#1a1f2a", fg=MUTED))
        self.auto_btn = PillButton(side, "自动定线（建议）", self.auto_threshold, variant="subtle", width=178, height=30, font=self.fonts.small, icon="✦")
        self.auto_btn.pack(padx=pad, pady=(8, 0))
        self.auto_suggestion: float | None = None
        self.threshold_note = tk.Label(side, text="分数越高越清晰；调整后立即重新分类", bg=PANEL, fg=FAINT, font=self.fonts.tiny, wraplength=176, justify="left", cursor="arrow")
        self.threshold_note.pack(anchor="w", padx=pad, pady=(5, 0))
        self.threshold_note.bind("<Button-1>", self._apply_auto_suggestion)

        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=pad, pady=(16, 12))
        tk.Label(side, text="排序", bg=PANEL, fg=FAINT, font=self.fonts.tiny).pack(anchor="w", padx=pad)
        self.sort_btn = tk.Label(side, text="导入顺序  ▾", bg="#1a1f2a", fg=TEXT_SOFT, font=self.fonts.small, padx=10, pady=7, anchor="w", cursor="hand2")
        sort_labels = {"import": "导入顺序", "score_asc": "清晰度 ↑", "score_desc": "清晰度 ↓", "name": "文件名", "date": "修改时间"}
        self.sort_btn.configure(text=f"{sort_labels.get(self.sort_mode, '导入顺序')}  ▾")
        self.sort_btn.pack(fill="x", padx=pad, pady=(4, 0))
        self.sort_btn.bind("<Button-1>", self._show_sort_menu)
        self.sort_btn.bind("<Enter>", lambda _e: self.sort_btn.configure(bg="#252d3c"))
        self.sort_btn.bind("<Leave>", lambda _e: self.sort_btn.configure(bg="#1a1f2a"))

        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=pad, pady=(14, 10))
        self.cache_label = tk.Label(side, text="", bg=PANEL, fg=FAINT, font=self.fonts.tiny, wraplength=176, justify="left")
        self.cache_label.pack(anchor="w", padx=pad)
        PillButton(side, "清理无效记录", self.cleanup_invalid_records, variant="subtle", width=178, height=28,
                   font=self.fonts.small).pack(padx=pad, pady=(7, 0))
        PillButton(side, "清空当前列表", self.clear_photo_list, variant="subtle", width=178, height=28,
                   font=self.fonts.small).pack(padx=pad, pady=(4, 0))
        self._update_cache_label()

        bottom = tk.Frame(side, bg=PANEL)
        bottom.pack(side="bottom", fill="x", padx=pad, pady=12)
        tk.Label(bottom, text="本地分析 · 不上传 · 删除只进回收站", bg=PANEL, fg=FAINT, font=self.fonts.tiny).pack(anchor="w")
        tk.Label(bottom, text=f"v{APP_VERSION}  ·  BlazeFace + Haar", bg=PANEL, fg="#414a5c", font=self.fonts.tiny).pack(anchor="w", pady=(2, 0))

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, bg=PANEL, height=34)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status_left = tk.Label(bar, text="", bg=PANEL, fg=MUTED, font=self.fonts.tiny)
        self.status_left.pack(side="left", padx=16)
        self.status_right = tk.Label(bar, text="← → 浏览   ·   空格 预览   ·   H 热力   ·   F 人脸   ·   G 焦点   ·   ? 帮助", bg=PANEL, fg=FAINT, font=self.fonts.tiny)
        self.status_right.pack(side="right", padx=16)

    # ------------------------------------------------------------------ keys
    def _main_key(self, event, callback) -> None:
        """Run a global shortcut only when the gallery owns the keyboard.

        Preview and settings are separate Toplevels with their own bindings;
        without this guard Delete/Esc/arrow keys would fire twice.
        """
        try:
            widget = self.root.focus_get()
            top = widget.winfo_toplevel() if widget is not None else None
        except (tk.TclError, AttributeError):
            top = None
        if top is not None and top is not self.root:
            return
        callback()

    def _bind_keys(self):
        def bind(seq: str, callback):
            self.root.bind_all(seq, lambda e, cb=callback: self._main_key(e, cb))

        bind("<Control-o>", self.add_files)
        bind("<Control-O>", self.add_folder)
        bind("<Control-e>", self.export_report)
        bind("<Control-a>", self.select_all)
        bind("<Left>", lambda: self.navigate(-1))
        bind("<Right>", lambda: self.navigate(1))
        bind("<Up>", lambda: self.navigate(-max(1, self.grid._cols)))
        bind("<Down>", lambda: self.navigate(max(1, self.grid._cols)))
        bind("<Home>", self.select_first)
        bind("<End>", self.select_last)
        bind("<Prior>", lambda: self.navigate(-max(1, self.grid._cols) * 3))
        bind("<Next>", lambda: self.navigate(max(1, self.grid._cols) * 3))
        bind("<space>", self.open_preview)
        bind("<Return>", self.open_preview)
        bind("<Delete>", self.delete_selected)
        bind("<F5>", self.rescan_selected)
        bind("<KeyPress-h>", self.toggle_heatmap)
        bind("<KeyPress-H>", self.toggle_heatmap)
        bind("<KeyPress-f>", self.toggle_faces)
        bind("<KeyPress-F>", self.toggle_faces)
        bind("<KeyPress-g>", self.toggle_focus)
        bind("<KeyPress-G>", self.toggle_focus)
        bind("<KeyPress-1>", lambda: self.set_filter("all"))
        bind("<KeyPress-2>", lambda: self.set_filter("sharp"))
        bind("<KeyPress-3>", lambda: self.set_filter("review"))
        bind("<KeyPress-4>", lambda: self.set_filter("blur"))
        bind("<KeyPress-5>", lambda: self.set_filter("error"))
        bind("<KeyPress-6>", lambda: self.set_filter("occluded"))
        bind("<KeyPress-7>", lambda: self.set_filter("duplicates"))
        bind("<KeyPress-plus>", lambda: self.set_threshold(self.threshold + 5))
        bind("<KeyPress-equal>", lambda: self.set_threshold(self.threshold + 5))
        bind("<KeyPress-minus>", lambda: self.set_threshold(self.threshold - 5))
        bind("<Escape>", self._on_escape)
        bind("<KeyPress-question>", self.show_help)
        if DND_AVAILABLE:
            try:
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind("<<Drop>>", self._on_drop)
                self.root.dnd_bind("<<DropEnter>>", lambda _e: self.grid.set_drop_active(True))
                self.root.dnd_bind("<<DropLeave>>", lambda _e: self.grid.set_drop_active(False))
            except Exception:
                globals()["DND_AVAILABLE"] = False

    # ------------------------------------------------------------------ import
    def add_files(self):
        types = [
            ("图片", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.heic *.heif"),
            ("HEIC/HEIF", "*.heic *.heif"),
            ("所有文件", "*.*"),
        ]
        paths = filedialog.askopenfilenames(title="添加照片", initialdir=self._last_dir, filetypes=types)
        if paths:
            self._last_dir = str(Path(paths[0]).parent)
            self._add_paths([Path(p) for p in paths], from_dialog=True)

    def add_folder(self):
        folder = filedialog.askdirectory(title="选择照片文件夹", initialdir=self._last_dir, mustexist=True)
        if folder:
            self._last_dir = folder
            self._add_paths([Path(folder)], from_dialog=True)

    def _on_drop(self, event):
        self.grid.set_drop_active(False)
        try:
            raw = self.root.tk.splitlist(event.data)
        except Exception:
            raw = [event.data]
        paths = [Path(p) for p in raw if str(p).strip()]
        self._add_paths(paths, from_dialog=False)

    @staticmethod
    def _photo_key(path: Path) -> str:
        try:
            return os.path.normcase(str(path.resolve()))
        except OSError:
            return os.path.normcase(os.path.abspath(str(path)))

    def _new_photo(self, path: Path, info: dict[str, Any] | None = None) -> Photo:
        mtime = 0.0
        size = 0
        try:
            st = path.stat()
            mtime, size = st.st_mtime, st.st_size
        except OSError:
            pass
        p = Photo(path=path, key=self._photo_key(path), name=path.name or str(path), mtime=mtime, size=size)
        if info and info.get("code") != "ok":
            p.result = make_error_result(path, info)
            p.state = "error"
        return p

    def _add_paths(self, paths: list[Path], from_dialog: bool = False):
        if not paths:
            return
        # Keep the UI responsive while a large folder is enumerated.  Reading
        # image headers is deferred to workers; only cheap explicit failures
        # (missing/empty/unsupported/HEIC-without-decoder) become error cards.
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            files, ignored, bad = collect_import_candidates(
                paths, recursive=bool(self.settings.get("recursive", True))
            )
        finally:
            self.root.configure(cursor="arrow")

        new: list[Photo] = []
        queued: list[Photo] = []
        known = set(self.by_key)
        for f in files:
            key = self._photo_key(f)
            if key in known:
                continue
            known.add(key)
            info = probe_image(f, deep=False)
            p = self._new_photo(f, info)
            new.append(p)
            if not p.is_error:
                queued.append(p)
        for f in bad:
            key = self._photo_key(f)
            if key in known:
                continue
            known.add(key)
            info = probe_image(f, deep=False)
            new.append(self._new_photo(f, info))

        if not new:
            if files or bad:
                self.toast.show("这些照片已经在列表里了")
            else:
                self.toast.show("没有找到可识别的图片" if not ignored else f"没有可识别的图片，已忽略 {ignored} 个文件")
            return
        for p in new:
            self.photos.append(p)
            self.by_key[p.key] = p
        first_existing = next((p for p in new if p.path.parent.exists()), new[0])
        self._last_dir = str(first_existing.path.parent)
        self.settings["last_dir"] = self._last_dir
        self._save_settings()
        if self.selected_key is None:
            self.selected_key = new[0].key
        if not self.selected_keys:
            self.selected_keys = {self.selected_key}
        self._cancel = False
        self._sort_frozen = self.sort_mode != "import"
        self.filter_mode = "all"
        self._queue_total += len(queued)
        if queued:
            self._active = True
            self._queue_done = max(0, self._queue_done)
        self._update_counts()
        self._refresh_visible()
        self._refresh_grid()
        if self.selected_key:
            self.select(self.selected_key)
        self.grid.canvas.yview_moveto(0)
        self._draw_progress()
        for p in queued:
            future = self.executor.submit(self._analyze_photo, p)
            with self._jobs_lock:
                self._analysis_jobs[future] = p.key
            future.add_done_callback(self._job_done)
        invalid = sum(1 for p in new if p.is_error)
        msg = f"已添加 {len(new)} 张照片"
        if invalid:
            msg += f" · {invalid} 张无法读取，已归入「异常」"
        if ignored:
            msg += f" · 忽略 {ignored} 个非图片文件"
        self.toast.show(msg, 3400)
        self._update_counts()

    def _analyze_photo(self, photo: Photo) -> tuple[str, dict]:
        try:
            return photo.key, self.analyzer.analyze(photo.path, {"faces": True, "occlusion": True})
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            import traceback

            log_error(traceback.format_exc())
            return photo.key, make_error_result(
                photo.path,
                {"code": "analysis_failed", "reason": f"分析失败：{exc}"},
            )

    def _post(self, callback, *args: Any) -> None:
        if self._closing:
            return
        self._callback_queue.put((callback, args))

    def _drain_callbacks(self) -> None:
        if self._closing:
            return
        try:
            while True:
                callback, args = self._callback_queue.get_nowait()
                try:
                    callback(*args)
                except Exception as exc:  # noqa: BLE001 - one bad callback must not stop the queue
                    import traceback

                    log_error(traceback.format_exc())
                    if not self._closing:
                        try:
                            self.toast.show(f"界面更新失败：{exc}", 4000)
                        except Exception:
                            pass
        except queue.Empty:
            pass
        finally:
            if not self._closing:
                try:
                    self._callback_job = self.root.after(40, self._drain_callbacks)
                except tk.TclError:
                    self._callback_job = None

    def _job_done(self, future: Future):
        if self._closing:
            return
        with self._jobs_lock:
            fallback_key = self._analysis_jobs.pop(future, "")
        try:
            key, result = future.result()
        except CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 - worker failure should not kill batch
            import traceback

            log_error(traceback.format_exc())
            key = fallback_key
            result = make_error_result(fallback_key, {"code": "analysis_failed", "reason": f"分析失败：{exc}"})
        self._post(self._on_analysis_result, key, result)

    def _on_analysis_result(self, key: str, result: dict):
        if self._closing:
            return
        trace = result.pop("error_trace", "")
        if trace:
            log_error(str(trace))
        if self._cancel:
            photo = self.by_key.get(key)
            if photo:
                photo.result = result
                photo.state = "error" if result.get("error") else "done"
                if not result.get("error"):
                    photo.cache_key = result.get("cache_key", "")
                    photo.thumb_path = self.analyzer.thumbnail_path(photo.cache_key) if photo.cache_key else None
                if key == self.selected_key:
                    self.inspector.update_result()
            self._schedule_refresh()
            self._update_counts()
            return
        self._queue_done += 1
        photo = self.by_key.get(key)
        if photo:
            photo.result = result
            photo.state = "error" if result.get("error") else "done"
            if not result.get("error"):
                photo.cache_key = result.get("cache_key", "")
                photo.thumb_path = self.analyzer.thumbnail_path(photo.cache_key) if photo.cache_key else None
            if key == self.selected_key:
                self.inspector.update_result()
        if self._queue_done >= self._queue_total:
            self._active = False
            self._sort_frozen = False
            self._queue_total = 0
            self._queue_done = 0
            self._rebuild_duplicates()
            if self.selected_key and self.selected_key in self.by_key:
                self.inspector.update_result()
            self._refresh_visible()
            self._refresh_grid()
            self._update_cache_label()
        else:
            self._schedule_refresh()
        self._update_counts()
        self._draw_progress()

    # ------------------------------------------------------------------ view state
    def _status_of(self, p: Photo) -> str:
        return p.status(self.threshold)

    def _passes_filter(self, p: Photo) -> bool:
        if self.filter_mode == "all":
            return True
        if self.filter_mode == "error":
            return p.is_error
        # Pending cards stay visible only in "all"; in a status filter they
        # would appear and disappear and break continuous preview navigation.
        if p.result is None:
            return False
        st = self._status_of(p)
        if self.filter_mode == "sharp":
            return st == "sharp"
        if self.filter_mode == "review":
            return st in ("review", "uncertain")
        if self.filter_mode == "blur":
            return st == "blur"
        if self.filter_mode == "occluded":
            return bool(p.result and p.result.get("occluded"))
        if self.filter_mode == "duplicates":
            return p.key in self.dup_of
        return True

    def _sort_photos(self, photos: list[Photo]) -> list[Photo]:
        mode = self.sort_mode
        if self._active and mode != "import":
            # Do not make cards jump while a scan is running.
            return list(photos)
        if mode == "name":
            return sorted(photos, key=lambda p: natural_key(p.name))
        if mode == "date":
            return sorted(photos, key=lambda p: p.mtime, reverse=True)
        if mode in ("score_asc", "score_desc"):
            done = [p for p in photos if p.result and not p.result.get("error")]
            pending = [p for p in photos if not (p.result and not p.result.get("error"))]
            done.sort(key=lambda p: (p.result.get("score", 0), natural_key(p.name)), reverse=(mode == "score_desc"))
            return done + pending
        return list(photos)

    def _rebuild_duplicates(self):
        records: list[dict[str, Any]] = []
        by_key: dict[str, Photo] = {}
        for p in self.photos:
            if not p.result or p.result.get("error") or not p.result.get("fingerprint"):
                continue
            record = dict(p.result)
            record["key"] = p.key
            records.append(record)
            by_key[p.key] = p
        groups: list[list[Photo]] = []
        for keys in find_similar_groups(records):
            group = [by_key[k] for k in keys if k in by_key]
            if len(group) > 1:
                groups.append(group)
        groups.sort(key=lambda g: (-len(g), g[0].name))
        self.dup_groups = groups
        self.dup_of = {}
        for idx, group in enumerate(groups):
            group.sort(key=lambda p: (-(p.result or {}).get("score", 0), -p.size))
            for p in group:
                self.dup_of[p.key] = idx
        self.dup_count = sum(len(g) for g in groups)

    def duplicate_group_for(self, key: str) -> list[Photo]:
        idx = self.dup_of.get(key)
        return self.dup_groups[idx] if idx is not None else []

    def _refresh_visible(self):
        photos = [p for p in self.photos if self._passes_filter(p)]
        self.visible_photos = self._sort_photos(photos)
        visible = {p.key for p in self.visible_photos}
        self.selected_keys &= visible
        if self.selected_key not in visible:
            self.selected_key = self.visible_photos[0].key if self.visible_photos else None
            self.selected_keys = {self.selected_key} if self.selected_key else set()
            if self.selected_key:
                self.inspector.show_photo(self.by_key[self.selected_key])
                self._request_preview(self.by_key[self.selected_key])
            else:
                self.inspector.clear()
        elif not self.selected_keys:
            self.selected_keys = {self.selected_key}
        try:
            keys = [p.key for p in self.visible_photos]
            self.inspector.nav_counter.configure(text=f"{keys.index(self.selected_key) + 1} / {len(keys)}")
        except (ValueError, AttributeError):
            pass
        self._update_batch_bar()

    def _refresh_grid(self):
        self._refresh_visible()
        self.grid.schedule_render()

    def _schedule_refresh(self):
        if self._refresh_job:
            return
        self._refresh_job = self.root.after(90, self._do_scheduled_refresh)

    def _do_scheduled_refresh(self):
        self._refresh_job = None
        self._refresh_visible()
        self.grid.schedule_render()
        self._update_counts()

    def _select_first(self):
        if self.visible_photos:
            self.select(self.visible_photos[0].key)

    def select_first(self):
        self._select_first()

    def select_last(self):
        if self.visible_photos:
            self.select(self.visible_photos[-1].key)

    def select(self, key: str, additive: bool = False, range_select: bool = False):
        photo = self.by_key.get(key)
        if not photo:
            return
        keys = [p.key for p in self.visible_photos]
        if range_select and self.selection_anchor in keys and key in keys:
            a, b = sorted((keys.index(self.selection_anchor), keys.index(key)))
            chosen = set(keys[a : b + 1])
            self.selected_keys = self.selected_keys | chosen if additive else chosen
        elif additive:
            if key in self.selected_keys:
                self.selected_keys.discard(key)
                if not self.selected_keys:
                    self.selected_key = None
                    self.inspector.clear()
                    self._refresh_grid()
                    self._update_statusbar()
                    return
                key = next(iter(self.selected_keys))
            else:
                self.selected_keys.add(key)
        else:
            self.selected_keys = {key}
        self.selected_key = key
        self.selection_anchor = key
        self._refresh_grid()
        self.inspector.show_photo(self.by_key[key])
        self._request_preview(self.by_key[key])
        self.grid.scroll_to_key(key)
        self._update_statusbar()
        self._update_batch_bar()

    def _update_batch_bar(self):
        n = len(self.selected_keys)
        if n > 1:
            self.batch_label.configure(text=f"已选 {n} 张")
            self.batch_bar.place(relx=0.5, rely=1.0, y=-16, anchor="s")
            self.batch_bar.lift()
        else:
            self.batch_bar.place_forget()

    def select_all(self):
        self.selected_keys = {p.key for p in self.visible_photos}
        if self.selected_key not in self.selected_keys:
            self.selected_key = self.visible_photos[0].key if self.visible_photos else None
        self._refresh_grid()
        self._update_statusbar()
        self._update_batch_bar()

    def clear_selection(self):
        self.selected_keys.clear()
        self.selected_key = None
        self._refresh_grid()
        self.inspector.clear()
        self._update_statusbar()
        self._update_batch_bar()

    def delete_selected(self):
        keys = list(self.selected_keys) or ([self.selected_key] if self.selected_key else [])
        self.delete_photos(keys, confirm=True)

    def delete_photos(self, keys: list[str], confirm: bool = True, parent=None) -> bool:
        photos = [self.by_key[k] for k in keys if k in self.by_key]
        if not photos:
            return False
        existing = [p for p in photos if p.path.is_file()]
        missing = [p for p in photos if not p.path.is_file()]
        if not existing:
            self.toast.show("这些文件已经不存在，无法移入回收站；可用「从列表移除」清理记录", 4200)
            return False
        if confirm:
            names = "\n".join(f"· {p.name}" for p in photos[:6])
            if len(photos) > 6:
                names += f"\n… 另有 {len(photos) - 6} 个文件"
            total = sum(p.size for p in existing)
            missing_note = f"\n其中 {len(missing)} 个文件已不存在，只会从列表清除。" if missing else ""
            ok = messagebox.askyesno(
                "移入回收站",
                f"将 {len(existing)} 个文件（{human_size(total)}）移入回收站？\n\n{names}\n\n"
                f"文件会进入 Windows 回收站，可在回收站恢复。程序不会直接永久删除。{missing_note}",
                icon="warning",
                parent=parent or self.root,
            )
            if not ok:
                return False
        success, error = recycle_files([p.path for p in existing], self.root.winfo_id())
        if not success:
            messagebox.showerror("删除失败", error, parent=parent or self.root)
            return False
        removed = {p.key for p in photos}
        self.photos = [p for p in self.photos if p.key not in removed]
        for key in removed:
            self.by_key.pop(key, None)
        self.selected_keys -= removed
        if self.selected_key in removed:
            self.selected_key = None
        if self.preview_window is not None:
            try:
                self.preview_window.refresh_after_removal()
            except Exception:
                pass
        self._rebuild_duplicates()
        self._refresh_visible()
        self._refresh_grid()
        self._update_counts()
        if self.selected_key:
            self.select(self.selected_key)
        elif self.visible_photos:
            self.select(self.visible_photos[0].key)
        else:
            self.inspector.clear()
        msg = f"已将 {len(existing)} 个文件移入回收站"
        if missing:
            msg += f"，并清除 {len(missing)} 个失效记录"
        self.toast.show(msg)
        return True

    def navigate(self, delta: int):
        if not self.visible_photos:
            return
        keys = [p.key for p in self.visible_photos]
        if self.selected_key in keys:
            idx = keys.index(self.selected_key)
        else:
            idx = 0
        idx = max(0, min(len(keys) - 1, idx + delta))
        self.select(keys[idx])

    def _on_escape(self):
        self.clear_selection()

    # ------------------------------------------------------------------ preview
    def _request_preview(self, photo: Photo):
        self._preview_token += 1
        token = self._preview_token
        self.preview_executor.submit(self._load_preview_task, token, photo.key, photo.path)

    def get_card_thumb(self, photo: Photo, w: int, h: int) -> ImageTk.PhotoImage | None:
        try:
            src = load_card_source(photo.path, photo.thumb_path)
            img = fit_on_card(src, (w, h))
            img = rounded_image(img, 10)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    def _load_preview_task(self, token: int, key: str, path: Path):
        if self._closing or token != self._preview_token:
            return
        try:
            full, preview = load_full_and_preview(path)
        except Exception:
            return
        self._post(self._preview_loaded, token, key, full, preview)

    def _preview_loaded(self, token: int, key: str, full: Image.Image, preview: Image.Image):
        if self._closing or token != self._preview_token or key != self.selected_key:
            return
        self.inspector.set_image(full, preview)
        self.inspector.update_result()

    def _preview_toggle(self, value: str):
        self.inspector._set_heat(value)

    def toggle_heatmap(self):
        self.inspector._set_heat("base" if self.inspector.heatmap else "heat")

    def toggle_faces(self):
        self.inspector._toggle_faces()

    def toggle_focus(self):
        self.inspector._toggle_focus()

    # ------------------------------------------------------------------ thresholds
    def _on_threshold(self, value: float):
        if self.auto_suggestion is not None:
            self._clear_auto_suggestion()
        self.threshold = float(value)
        self.threshold_label.configure(text=f"{self.threshold:.0f}")
        if hasattr(self, "slider") and abs(self.slider.value - self.threshold) > 0.01:
            self.slider.set_value(self.threshold)
        self.settings["threshold"] = self.threshold
        self._refresh_visible()
        self._refresh_grid()
        self._update_counts()
        if self.inspector.photo:
            self.inspector._update_summary()
            self.inspector._update_face_section()
        self._save_settings()

    def set_threshold(self, value: float):
        value = max(0, min(100, float(value)))
        self.slider.set_value(value)
        self._on_threshold(value)

    def auto_threshold(self):
        scores = [p.result["score"] for p in self.photos if p.result and not p.result.get("error")]
        if len(scores) < 4:
            self.toast.show("至少分析 4 张照片后才能建议判定线")
            return
        low, high = min(scores), max(scores)
        # Otsu on a 0..100 histogram
        hist = np.zeros(101, np.int32)
        for s in scores:
            hist[int(max(0, min(100, round(s))))] += 1
        total = hist.sum()
        sum_all = float(np.dot(np.arange(101), hist))
        wB = 0.0
        sumB = 0.0
        best = 60.0
        best_var = -1.0
        for t in range(5, 96):
            wB += hist[t]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += t * hist[t]
            mB = sumB / wB
            mF = (sum_all - sumB) / wF
            var = wB * wF * (mB - mF) ** 2
            if var > best_var:
                best_var = var
                best = float(t)
        if high - low < 8:
            best = 60.0
        best = max(35, min(80, best))
        # Suggestion only.  Threshold changes must always be the user's choice.
        self.auto_suggestion = best
        self.threshold_note.configure(
            text=f"本批建议判定线 {best:.0f}（仅建议）\n点击这里应用",
            fg=ACCENT,
            cursor="hand2",
        )
        self.toast.show(f"建议判定线 {best:.0f}，尚未修改；点击左侧建议文字可应用", 3800)

    def _apply_auto_suggestion(self, _event=None) -> None:
        if self.auto_suggestion is None:
            return
        value = self.auto_suggestion
        self.auto_suggestion = None
        self.threshold_note.configure(
            text="分数越高越清晰；调整后立即重新分类",
            fg=FAINT,
            cursor="arrow",
        )
        self.set_threshold(value)

    def _clear_auto_suggestion(self) -> None:
        self.auto_suggestion = None
        try:
            self.threshold_note.configure(
                text="分数越高越清晰；调整后立即重新分类",
                fg=FAINT,
                cursor="arrow",
            )
        except tk.TclError:
            pass

    def save_preview_prefs(self):
        self.settings["preview_heatmap"] = bool(self.inspector.heatmap)
        self.settings["preview_faces"] = bool(self.inspector.show_faces)
        self.settings["preview_focus"] = bool(self.inspector.show_focus)
        self._save_settings()

    # ------------------------------------------------------------------ filters/sort
    def set_filter(self, mode: str):
        if mode not in self.filter_rows:
            return
        self.filter_mode = mode
        self.settings["filter"] = mode
        self._update_counts()
        self._refresh_visible()
        self._refresh_grid()
        self._save_settings()

    def _update_counts(self):
        counts = {
            "all": len(self.photos),
            "sharp": 0,
            "review": 0,
            "blur": 0,
            "error": 0,
            "occluded": 0,
            "duplicates": self.dup_count,
        }
        errors = 0
        pending = 0
        for p in self.photos:
            st = self._status_of(p)
            if st == "sharp":
                counts["sharp"] += 1
            elif st in ("review", "uncertain"):
                counts["review"] += 1
            elif st == "blur":
                counts["blur"] += 1
            elif st == "error":
                errors += 1
                counts["error"] += 1
            elif st == "pending":
                pending += 1
            if p.result and p.result.get("occluded"):
                counts["occluded"] += 1
        for key, (row, lab, cnt, bar) in self.filter_rows.items():
            active = key == self.filter_mode
            bg = CARD_HOVER if active else PANEL
            row.configure(bg=bg)
            for child in row.winfo_children():
                try:
                    child.configure(bg=bg)
                except tk.TclError:
                    pass
            bar.configure(bg=ACCENT if active else bg)
            lab.configure(fg=TEXT if active else TEXT_SOFT, font=self.fonts.body_bold if active else self.fonts.body)
            cnt.configure(text=str(counts.get(key, 0)), fg=ACCENT if active else FAINT)
        bits = [f"共 {counts['all']} 张", f"清晰 {counts['sharp']}", f"待复核 {counts['review']}", f"模糊 {counts['blur']}"]
        if counts["occluded"]:
            bits.append(f"遮挡 {counts['occluded']}")
        if pending:
            bits.append(f"分析中 {pending}")
        if errors:
            bits.append(f"异常 {errors}")
        self.status_left.configure(text="   ·   ".join(bits))
        self._update_statusbar()
        self._draw_progress()

    def _update_statusbar(self):
        if self._hover_path:
            self.status_right.configure(text=ellipsize(self._hover_path, self.font_obj(self.fonts.tiny), 760))
        elif len(self.selected_keys) > 1:
            self.status_right.configure(text=f"已选 {len(self.selected_keys)} 张   ·   Delete 移入回收站   ·   Esc 取消选择")
        else:
            hint = "← → 浏览   ·   空格 预览   ·   H 热力   ·   F 人脸   ·   G 焦点   ·   ? 帮助"
            self.status_right.configure(text=hint)

    def set_hover_path(self, photo: Photo | None):
        self._hover_path = str(photo.path) if photo else None
        self._update_statusbar()

    def _show_sort_menu(self, event=None):
        menu = tk.Menu(self.root, tearoff=0, bg=PANEL_2, fg=TEXT_SOFT, activebackground=CARD_PRESSED,
                       activeforeground=TEXT, bd=0, font=self.fonts.small)
        options = [
            ("import", "导入顺序"),
            ("score_asc", "清晰度 ↑ 先看可疑"),
            ("score_desc", "清晰度 ↓ 先看最佳"),
            ("name", "文件名"),
            ("date", "修改时间（新→旧）"),
        ]
        for key, label in options:
            menu.add_radiobutton(label=label, value=key, variable=tk.StringVar(value=self.sort_mode),
                                 command=lambda k=key: self.set_sort(k))
        menu.tk_popup(self.sort_btn.winfo_rootx(), self.sort_btn.winfo_rooty() + self.sort_btn.winfo_height())

    def set_sort(self, mode: str):
        self.sort_mode = mode
        labels = {"import": "导入顺序", "score_asc": "清晰度 ↑", "score_desc": "清晰度 ↓", "name": "文件名", "date": "修改时间"}
        self.sort_btn.configure(text=f"{labels.get(mode, mode)}  ▾")
        self.settings["sort"] = mode
        self._refresh_visible()
        self._refresh_grid()
        self._save_settings()

    # ------------------------------------------------------------------ actions
    def open_original(self, key: str | None = None):
        key = key or self.selected_key
        p = self.by_key.get(key) if key else None
        if not p:
            return
        try:
            os.startfile(str(p.path))
        except Exception as exc:
            self.toast.show(f"无法打开：{exc}")

    def open_selected(self):
        self.open_preview()

    def open_external(self, key: str | None = None):
        self.open_original(key)

    def open_preview(self, key: str | None = None):
        if key:
            self.select(key)
        if not self.visible_photos:
            self.toast.show("先导入照片，再打开大图预览")
            return
        if not self.selected_key or self.selected_key not in self.by_key:
            self.select(self.visible_photos[0].key)
        if self.preview_window is not None:
            try:
                if self.preview_window.winfo_exists():
                    self.preview_window.focus_force()
                    return
            except tk.TclError:
                pass
        self.preview_window = LargePreview(self.root, self)

    def on_preview_closed(self):
        self.preview_window = None
        if self.selected_key and self.selected_key in self.by_key:
            self.select(self.selected_key)

    def show_duplicate_dialog(self):
        if not self.dup_groups:
            self.toast.show("没有检测到同图的不同格式版本")
            return
        DuplicateDialog(self.root, self)

    def show_context_menu(self, x: int, y: int):
        p = self.by_key.get(self.selected_key) if self.selected_key else None
        if not p:
            return
        menu = tk.Menu(self.root, tearoff=0, bg=PANEL_2, fg=TEXT_SOFT, activebackground=CARD_PRESSED,
                       activeforeground=TEXT, bd=0, font=self.fonts.small)
        menu.add_command(label="大图预览", command=self.open_preview)
        menu.add_command(label="用系统应用打开", command=self.open_external)
        menu.add_command(label="在文件夹中显示", command=lambda: self.reveal(p))
        menu.add_command(label="复制文件路径", command=lambda: self.copy_path(p))
        menu.add_separator()
        if p.is_error:
            menu.add_command(label="重试读取", command=lambda: self.retry_photo(p.key))
            menu.add_command(label="从列表移除（不删除文件）", command=lambda: self.remove_photo(p.key))
        else:
            menu.add_command(label="移入回收站（可恢复）", command=self.delete_selected)
            menu.add_command(label="重新分析", command=lambda: self.retry_photo(p.key))
            menu.add_command(label="从列表移除（不删除文件）", command=self.remove_selected)
        menu.tk_popup(x, y)

    def reveal(self, p: Photo):
        parent = p.path.parent
        try:
            if p.path.exists():
                subprocess.Popen(["explorer.exe", "/select,", str(p.path)])
            elif parent.exists():
                os.startfile(str(parent))
            else:
                self.toast.show("文件所在文件夹也不存在了")
        except Exception as exc:
            self.toast.show(f"无法打开文件夹：{exc}")

    def copy_path(self, p: Photo):
        self.root.clipboard_clear()
        self.root.clipboard_append(str(p.path))
        self.toast.show("路径已复制")

    def remove_photo(self, key: str) -> None:
        p = self.by_key.get(key)
        if not p:
            return
        self.photos = [x for x in self.photos if x.key != key]
        self.by_key.pop(key, None)
        self.selected_keys.discard(key)
        if self.preview_window is not None:
            try:
                self.preview_window.refresh_after_removal()
            except Exception:
                pass
        if self.selected_key == key:
            self.selected_key = None
        self._rebuild_duplicates()
        self._refresh_visible()
        self._refresh_grid()
        self._update_counts()
        if not self.visible_photos:
            self.inspector.clear()
        elif self.selected_key:
            self.select(self.selected_key)
        else:
            self.select(self.visible_photos[0].key)
        self.toast.show("已从列表移除，原文件未删除")

    def remove_selected(self):
        if self.selected_key:
            self.remove_photo(self.selected_key)

    def retry_photo(self, key: str | None = None) -> None:
        key = key or self.selected_key
        p = self.by_key.get(key) if key else None
        if not p:
            return
        options = {"faces": True, "occlusion": True}
        self.analyzer.invalidate(p.path, options)
        p.result = None
        p.state = "pending"
        p.cache_key = ""
        p.thumb_path = None
        self._cancel = False
        self._sort_frozen = self.sort_mode != "import"
        self._queue_total += 1
        self._active = True
        self._refresh_grid()
        self._update_counts()
        self._draw_progress()
        self.inspector.update_result()
        future = self.executor.submit(self._analyze_photo, p)
        with self._jobs_lock:
            self._analysis_jobs[future] = p.key
        future.add_done_callback(self._job_done)
        self.toast.show(f"正在重试 {p.name}")

    def rescan_selected(self):
        p = self.by_key.get(self.selected_key) if self.selected_key else None
        targets = [p] if p else [x for x in self.photos if not x.is_error]
        if not targets:
            return
        self._cancel = False
        self._sort_frozen = True
        self._queue_total += len(targets)
        self._active = True
        for photo in targets:
            self.analyzer.invalidate(photo.path, {"faces": True, "occlusion": True})
            photo.result = None
            photo.state = "pending"
            photo.cache_key = ""
            photo.thumb_path = None
            future = self.executor.submit(self._analyze_photo, photo)
            with self._jobs_lock:
                self._analysis_jobs[future] = photo.key
            future.add_done_callback(self._job_done)
        self._refresh_grid()
        self._update_counts()
        self._draw_progress()
        self.toast.show(f"正在重新分析 {len(targets)} 张")

    def add_face_tip(self, face: dict):
        pass

    def select_face_tip(self, face: dict):
        pass

    # ------------------------------------------------------------------ session / settings
    def restore_last_session(self) -> tuple[list[Path], int, int]:
        """Validate last_files, retire dead entries and enforce the cap.

        Returns ``(valid_paths, invalid_count, limited_count)``.  Only files
        that still exist, are readable and pass a header check are restored.
        """
        raw: list[str] = []
        for item in self.settings.get("last_files", []) or []:
            try:
                raw.append(str(Path(item)))
            except (TypeError, ValueError):
                continue
        if not raw:
            return [], 0, 0
        valid, invalid, limited = validate_restore_candidates(raw, MAX_RESTORE_FILES)
        self.settings["last_files"] = [str(p) for p in valid]
        # Use the raw writer here: _save_settings() writes the current photo
        # list, which is intentionally still empty while the window starts.
        save_settings(self.settings)
        return valid, invalid, limited

    def cleanup_invalid_records(self) -> int:
        """Remove missing/unreadable/unsupported entries from the current list."""
        if not self.photos:
            self.toast.show("当前列表为空")
            return 0
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        removed: list[Photo] = []
        try:
            for p in list(self.photos):
                info = probe_image(p.path, deep=False)
                if info.get("code") != "ok":
                    removed.append(p)
        finally:
            self.root.configure(cursor="arrow")
        if not removed:
            self.toast.show("没有发现失效记录")
            return 0
        removed_keys = {p.key for p in removed}
        self.photos = [p for p in self.photos if p.key not in removed_keys]
        for key in removed_keys:
            self.by_key.pop(key, None)
        self.selected_keys -= removed_keys
        if self.selected_key in removed_keys:
            self.selected_key = None
        if self.preview_window is not None:
            try:
                self.preview_window.refresh_after_removal()
            except Exception:
                pass
        self._rebuild_duplicates()
        self._refresh_visible()
        self._refresh_grid()
        self._update_counts()
        if not self.visible_photos:
            self.inspector.clear()
        elif self.selected_key:
            self.select(self.selected_key)
        self._save_settings()
        self.toast.show(f"已清理 {len(removed)} 个失效记录，原文件不会被动")
        return len(removed)

    def clear_photo_list(self) -> None:
        if not self.photos:
            self.toast.show("当前列表已经是空的")
            return
        dialog_parent = self.settings_window if getattr(self, "settings_window", None) is not None else self.root
        try:
            if not dialog_parent.winfo_exists():
                dialog_parent = self.root
        except tk.TclError:
            dialog_parent = self.root
        if not messagebox.askyesno(
            "清空当前列表",
            "从当前列表移除全部照片？\n\n这只清空列表，不会删除、移动或修改任何原照片。\n"
            "下次启动也不会自动恢复这批照片。",
            icon="question",
            parent=dialog_parent,
        ):
            return
        if self._active:
            self.cancel_analysis()
        if self.preview_window is not None:
            try:
                self.preview_window.close()
            except Exception:
                pass
        self.photos.clear()
        self.by_key.clear()
        self.selected_keys.clear()
        self.selected_key = None
        self.visible_photos = []
        self.dup_groups = []
        self.dup_of = {}
        self.dup_count = 0
        self.settings["last_files"] = []
        self._save_settings()
        self.inspector.clear()
        self._refresh_visible()
        self._refresh_grid()
        self._update_counts()
        self._update_cache_label()
        self.toast.show("已清空当前列表，原照片仍在原处")

    def show_settings(self) -> None:
        win = getattr(self, "settings_window", None)
        if win is not None:
            try:
                if win.winfo_exists():
                    win.focus_force()
                    return
            except tk.TclError:
                pass
        self.settings_window = SettingsWindow(self.root, self)

    def cache_summary_text(self) -> str:
        stats = self.analyzer.cache_stats()
        return (
            f"缓存位置：{CACHE_DIR}\n"
            f"{stats['entries']} 条分析记录 · {stats['thumbs']} 个缩略图 · {human_size(stats['bytes'])}\n"
            f"其中 {stats['missing_sources']} 条源文件已经不存在。\n"
            "清理缓存不会删除或修改任何原照片。"
        )

    def purge_cache(self, mode: str, days: int = DEFAULT_CACHE_DAYS) -> None:
        labels = {
            "missing": "清理源文件已不存在的缓存",
            "old": f"清理 {days} 天未使用的缓存",
            "all": "清空全部分析缓存",
        }
        label = labels.get(mode, "清理缓存")
        if not messagebox.askyesno(
            label,
            f"{label}？\n\n缓存只包含缩略图和分析结果，不会影响原照片。\n"
            "下次导入需要重新分析，可能较慢。",
            icon="question",
            parent=getattr(self, "settings_window", None) or self.root,
        ):
            return
        if mode == "all":
            result = self.analyzer.purge_cache(remove_all=True, source_missing=True)
        elif mode == "old":
            result = self.analyzer.purge_cache(older_than_days=days, source_missing=True)
        else:
            result = self.analyzer.purge_cache(source_missing=True)
        self._update_cache_label()
        self.toast.show(f"已清理 {result['removed']} 个缓存文件，释放 {human_size(result['bytes'])}，原图未动")
        win = getattr(self, "settings_window", None)
        if win is not None and getattr(win, "refresh_stats", None):
            win.refresh_stats()

    # ------------------------------------------------------------------ export
    def export_report(self):
        photos = list(self.visible_photos)
        if not photos:
            self.toast.show("当前筛选结果为空")
            return
        default = f"FocusSort_报告_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        path = filedialog.asksaveasfilename(
            title="导出当前筛选列表",
            initialdir=self._last_dir,
            initialfile=default,
            defaultextension=".csv",
            filetypes=[("CSV 表格", "*.csv")],
        )
        if not path:
            return
        try:
            import csv

            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "文件名", "路径", "当前状态", "清晰度", "全局分", "人脸分", "人数",
                    "有遮挡", "遮挡说明", "人脸明细", "异常代码", "异常原因", "宽", "高",
                    "文件大小", "修改时间", "拉普拉斯方差", "细节P90", "频谱斜率",
                    "边缘密度", "置信度", "低细节",
                ])
                for p in photos:
                    r = p.result or {}
                    faces = r.get("faces", [])
                    face_bits = []
                    for i, face in enumerate(faces):
                        occ = face.get("occlusion", {})
                        face_bits.append(f"人脸{i+1}: {face.get('score', 0):.0f}分/{occ.get('label', '')}")
                    writer.writerow([
                        p.name,
                        str(p.path),
                        STATUS_LABELS.get(p.status(self.threshold), ""),
                        f"{r.get('score', 0):.1f}" if r else "",
                        f"{r.get('global_score', 0):.1f}" if r else "",
                        f"{r.get('face_score'):.1f}" if r.get("face_score") is not None else "",
                        r.get("face_count", 0),
                        "是" if r.get("occluded") else "否",
                        "；".join(r.get("occlusion_reasons", [])),
                        "；".join(face_bits),
                        r.get("error_code", ""),
                        r.get("error", "") if r.get("error") else "",
                        r.get("width", ""),
                        r.get("height", ""),
                        p.size,
                        format_date(p.mtime),
                        r.get("metrics", {}).get("lap_var", ""),
                        r.get("metrics", {}).get("detail_p90", ""),
                        r.get("metrics", {}).get("spectral_slope", ""),
                        r.get("metrics", {}).get("edge_density", ""),
                        r.get("confidence", ""),
                        "是" if r.get("low_detail") else "否",
                    ])
            self.toast.show(f"已导出 {len(photos)} 条到 {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    # ------------------------------------------------------------------ help
    def show_help(self):
        HelpWindow(self.root, self)

    # ------------------------------------------------------------------ progress
    def _draw_progress(self):
        c = self.progress_line
        if not hasattr(c, "winfo_width"):
            return
        try:
            w = max(1, c.winfo_width())
        except tk.TclError:
            return
        c.delete("all")
        c.create_rectangle(0, 0, w, 3, fill=BG, outline="")
        if self._active and self._queue_total:
            frac = max(0.0, min(1.0, self._queue_done / max(1, self._queue_total)))
            c.create_rectangle(0, 0, int(w * frac), 3, fill=ACCENT, outline="")
            self.top_info.configure(text=f"正在分析  {self._queue_done}/{self._queue_total}")
            self.stop_btn.pack(side="left", padx=(10, 0))
        else:
            self.stop_btn.pack_forget()
            if self.photos:
                analyzed = sum(1 for p in self.photos if p.result)
                self.top_info.configure(text=f"{len(self.photos)} 张照片  ·  已分析 {analyzed}")
            else:
                self.top_info.configure(text="把照片拖进来，或从右上角添加")

    def cancel_analysis(self):
        self._cancel = True
        cancelled = 0
        with self._jobs_lock:
            jobs = list(self._analysis_jobs.items())
        for fut, key in jobs:
            if not fut.done() and fut.cancel():
                photo = self.by_key.get(key)
                if photo and photo.result is None:
                    photo.state = "cancelled"
                    cancelled += 1
        self._active = False
        self._sort_frozen = False
        self._queue_total = 0
        self._queue_done = 0
        self._update_counts()
        self._refresh_grid()
        self._draw_progress()
        self.toast.show(f"已停止分析，已保留完成的结果" + (f"，{cancelled} 张取消" if cancelled else ""), 3600)

    def _report_callback_exception(self, exc_type, exc_value, tb):
        import traceback

        text = "".join(traceback.format_exception(exc_type, exc_value, tb))
        log_error(text)
        try:
            self.toast.show("出现了一个错误，已记录到 error.log", 4000)
        except Exception:
            pass

    def _startup_note(self):
        if not BLAZE_MODEL.exists():
            self.toast.show("未找到人脸模型，已使用备用检测")
        if not self.photos:
            self.toast.show("把照片或文件夹拖进窗口即可开始", 3400)

    def _update_cache_label(self):
        try:
            stats = self.analyzer.cache_stats()
            extra = f" · {stats['missing_sources']} 条失效" if stats["missing_sources"] else ""
            self.cache_label.configure(text=f"本地缓存 {stats['entries']} 张{extra}\n重复导入会直接读取结果")
        except Exception:
            self.cache_label.configure(text="本地缓存已启用")

    # ------------------------------------------------------------------ geometry/save
    def _save_settings(self):
        try:
            geo = self.root.geometry()
            if geo:
                self.settings["geometry"] = geo.split("+")[0]
        except tk.TclError:
            pass
        self.settings["threshold"] = self.threshold
        self.settings["sort"] = self.sort_mode
        self.settings["filter"] = self.filter_mode
        self.settings["last_dir"] = self._last_dir
        self.settings["last_files"] = [str(p.path) for p in self.photos[:MAX_RESTORE_FILES]]
        save_settings(self.settings)

    def on_close(self):
        self._closing = True
        self._save_settings()
        try:
            with self._jobs_lock:
                jobs = list(self._analysis_jobs)
            for fut in jobs:
                fut.cancel()
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.preview_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass


# ---------------------------------------------------------------------------
# large preview page
# ---------------------------------------------------------------------------
class LargePreview(tk.Toplevel):
    """A full-window preview page: big image, continuous flip, 1:1, overlays."""

    def __init__(self, parent, app: FocusSortApp):
        super().__init__(parent)
        self.app = app
        self.photos = list(app.visible_photos)
        self.index = 0
        keys = [p.key for p in self.photos]
        if app.selected_key in keys:
            self.index = keys.index(app.selected_key)
        self.current: Photo | None = None
        self.preview: Image.Image | None = None
        self.full: Image.Image | None = None
        self.preview_cache: OrderedDict[str, Image.Image] = OrderedDict()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="focus-preview-page")
        self._token = 0
        self._closing = False
        self._callback_queue: queue.Queue[tuple[Any, tuple]] = queue.Queue()
        self._poll_job = None
        self._heat_cache = None
        self._heat_key = None
        self._overlay_cache: OrderedDict[tuple, Image.Image] = OrderedDict()
        self.heat = bool(app.settings.get("preview_heatmap", False))
        self.faces = bool(app.settings.get("preview_faces", False))
        self.focus = bool(app.settings.get("preview_focus", False))
        self.zoom = 1.0
        self.pan = [0, 0]
        self._drag = None
        self._tk_img = None
        self._pending_one_to_one = False
        self.pixel_1to1 = False
        self._last_scale = 1.0

        self.title("大图预览 · FocusSort")
        self.configure(bg=BLACK)
        self.transient(parent)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{max(1100, sw - 80)}x{max(720, sh - 100)}+40+30")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._build()

        def key(seq: str, callback):
            # Returning "break" stops the main window's bind_all shortcuts
            # from also handling Delete/Esc/arrows for the gallery behind.
            self.bind(seq, lambda _e, cb=callback: (cb(), "break")[1])

        key("<Escape>", self.close)
        key("<Left>", lambda: self.go(-1))
        key("<Right>", lambda: self.go(1))
        key("<Home>", lambda: self.go_to(0))
        key("<End>", lambda: self.go_to(len(self.photos) - 1))
        key("<KeyPress-h>", self.toggle_heat)
        key("<KeyPress-H>", self.toggle_heat)
        key("<KeyPress-f>", self.toggle_faces)
        key("<KeyPress-F>", self.toggle_faces)
        key("<KeyPress-g>", self.toggle_focus)
        key("<KeyPress-G>", self.toggle_focus)
        key("<Delete>", self.delete_current)
        key("<Return>", self.one_to_one)
        key("<space>", self.one_to_one)
        key("<plus>", lambda: self.zoom_by(1.25))
        key("<minus>", lambda: self.zoom_by(0.8))
        self.after(100, self._show_index)
        self.after(600, self._tick)
        self._poll_job = self.after(40, self._drain_callbacks)
        self.focus_force()

    def _build(self):
        top = tk.Frame(self, bg="#0b0d12", height=56)
        top.pack(fill="x")
        top.pack_propagate(False)
        self.title_label = tk.Label(top, text="", bg="#0b0d12", fg=TEXT, font=self.app.fonts.h3, anchor="w")
        self.title_label.pack(side="left", padx=(18, 10))
        self.meta_label = tk.Label(top, text="", bg="#0b0d12", fg=MUTED, font=self.app.fonts.small, anchor="w")
        self.meta_label.pack(side="left")
        self.counter_label = tk.Label(top, text="", bg="#0b0d12", fg=TEXT_SOFT, font=self.app.fonts.small)
        self.counter_label.pack(side="right", padx=(10, 8))
        IconButton(top, "✕", self.close, size=34, tooltip="关闭 (Esc)", bg="#0b0d12", fg=TEXT_SOFT).pack(side="right", padx=(0, 12))

        self.canvas = tk.Canvas(self, bg=BLACK, highlightthickness=0, bd=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<ButtonPress-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_drag", None))
        self.canvas.bind("<Double-Button-1>", lambda _e: self.toggle_fit_100())

        bottom = tk.Frame(self, bg="#0b0d12", height=62)
        bottom.pack(fill="x")
        bottom.pack_propagate(False)
        left = tk.Frame(bottom, bg="#0b0d12")
        left.pack(side="left", padx=12, pady=12)
        PillButton(left, "‹ 上一张", lambda: self.go(-1), variant="ghost", width=98, height=32, font=self.app.fonts.small).pack(side="left", padx=3)
        PillButton(left, "下一张 ›", lambda: self.go(1), variant="ghost", width=98, height=32, font=self.app.fonts.small).pack(side="left", padx=3)
        PillButton(left, "适应", self.fit, variant="subtle", width=58, height=32, font=self.app.fonts.small).pack(side="left", padx=3)
        PillButton(left, "1:1", self.one_to_one, variant="subtle", width=52, height=32, font=self.app.fonts.small).pack(side="left", padx=3)
        self.heat_btn = PillButton(left, f"热力：{'开' if self.heat else '关'}", self.toggle_heat,
                                   variant="ghost" if self.heat else "subtle", width=82, height=32, font=self.app.fonts.small)
        self.heat_btn.pack(side="left", padx=3)
        self.face_btn = PillButton(left, f"人脸：{'开' if self.faces else '关'}", self.toggle_faces,
                                   variant="ghost" if self.faces else "subtle", width=82, height=32, font=self.app.fonts.small)
        self.face_btn.pack(side="left", padx=3)
        self.focus_btn = PillButton(left, f"焦点：{'开' if self.focus else '关'}", self.toggle_focus,
                                    variant="ghost" if self.focus else "subtle", width=82, height=32, font=self.app.fonts.small)
        self.focus_btn.pack(side="left", padx=3)
        right = tk.Frame(bottom, bg="#0b0d12")
        right.pack(side="right", padx=12, pady=12)
        PillButton(right, "重试读取", self.retry_current, variant="ghost", width=82, height=32, font=self.app.fonts.small).pack(side="left", padx=3)
        PillButton(right, "系统打开", self.external, variant="subtle", width=82, height=32, font=self.app.fonts.small).pack(side="left", padx=3)
        PillButton(right, "移入回收站", self.delete_current, variant="danger", width=100, height=32, font=self.app.fonts.small).pack(side="left", padx=3)
        tk.Label(bottom, text="← → 翻阅  ·  滚轮缩放  ·  双击 1:1/适应  ·  H 热力  ·  F 人脸  ·  G 焦点",
                 bg="#0b0d12", fg=FAINT, font=self.app.fonts.tiny).pack(side="right", padx=(0, 12))

    # ------------------------------------------------------------------ data
    def _show_index(self):
        if self._closing:
            return
        if not self.photos:
            self.close()
            return
        self.index = max(0, min(len(self.photos) - 1, self.index))
        self.current = self.photos[self.index]
        self.zoom = 1.0
        self.pan = [0, 0]
        self._pending_one_to_one = False
        self.pixel_1to1 = False
        self._had_result = bool(self.current.result)
        self._update_header()
        if self.current.is_error:
            self.preview = None
            self.full = None
            self._redraw()
            self._select_current()
            return
        cached = self.preview_cache.get(self.current.key)
        if cached is not None:
            self.preview = cached
            self.full = None
            self._redraw()
        else:
            self.preview = None
            self.full = None
            self._redraw()
            self._request(self.current, full=True)
        # Prefetch neighbours' screen previews.
        for offset in (-1, 1):
            idx = (self.index + offset) % len(self.photos)
            p = self.photos[idx]
            if p.key not in self.preview_cache:
                self._request(p, full=False)

    def _request(self, photo: Photo, full: bool):
        key = photo.key
        token = self._token
        self._executor.submit(self._load_task, token, key, photo.path, full)

    def _post(self, callback, *args: Any) -> None:
        if not self._closing:
            self._callback_queue.put((callback, args))

    def _drain_callbacks(self) -> None:
        if self._closing:
            return
        try:
            while True:
                callback, args = self._callback_queue.get_nowait()
                try:
                    callback(*args)
                except Exception:
                    import traceback

                    log_error(traceback.format_exc())
        except queue.Empty:
            pass
        finally:
            if not self._closing:
                try:
                    self._poll_job = self.after(40, self._drain_callbacks)
                except tk.TclError:
                    self._poll_job = None

    def _load_task(self, token: int, key: str, path: Path, want_full: bool):
        try:
            if want_full:
                full, preview = load_full_and_preview(path)
            else:
                with Image.open(path) as im:
                    im.draft("RGB", (2400, 2400))
                    im = ImageOps.exif_transpose(im)
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    im.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
                    preview = im.copy()
                full = None
        except Exception:
            return
        self._post(self._loaded, token, key, full, preview)

    def _loaded(self, token: int, key: str, full, preview):
        if self._closing:
            return
        self.preview_cache[key] = preview
        self.preview_cache.move_to_end(key)
        while len(self.preview_cache) > 6:
            self.preview_cache.popitem(last=False)
        if self.current and key == self.current.key:
            self.preview = preview
            if full is not None:
                self.full = full
            self._redraw()
            if self._pending_one_to_one and full is not None:
                self._pending_one_to_one = False
                self.one_to_one()

    # ------------------------------------------------------------------ header
    def _update_header(self):
        if not self.current:
            return
        p = self.current
        self.title_label.configure(text=p.name)
        self.counter_label.configure(text=f"{self.index + 1} / {len(self.photos)}")
        r = p.result or {}
        if r.get("error"):
            meta = f"无法读取  ·  {r.get('error', '')}"
        elif not r:
            meta = "已取消，可重新分析" if p.state == "cancelled" else "分析中…"
        else:
            status = p.status(self.app.threshold)
            meta = f"{STATUS_LABELS.get(status, '')}  ·  清晰度 {r.get('score', 0):.0f}  ·  {r.get('width', 0)}×{r.get('height', 0)}"
            if r.get("occluded"):
                meta += "  ·  面部疑似遮挡"
            focus = r.get("focus") or {}
            if focus:
                meta += "  ·  估计焦点：" + ("人脸" if focus.get("source") == "face" else "细节")
        self.meta_label.configure(text=meta, fg=STATUS_COLORS.get(p.status(self.app.threshold), MUTED))

    def _tick(self):
        if self._closing:
            return
        if self.current:
            self._update_header()
            if self.current.result and not getattr(self, "_had_result", False):
                self._had_result = True
                self._redraw()
        self.after(800, self._tick)

    # ------------------------------------------------------------------ display
    def _fit_scale(self, iw: int, ih: int, cw: int, ch: int) -> float:
        return max(0.01, min((cw - 48) / max(1, iw), (ch - 48) / max(1, ih)))

    def _heat_overlay(self, img: Image.Image) -> Image.Image:
        try:
            arr = np.asarray(img.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            mean = cv2.boxFilter(lap, -1, (25, 25), normalize=True, borderType=cv2.BORDER_REFLECT)
            m2 = cv2.boxFilter(lap * lap, -1, (25, 25), normalize=True, borderType=cv2.BORDER_REFLECT)
            var = np.maximum(m2 - mean * mean, 0)
            var = cv2.GaussianBlur(np.log1p(var), (0, 0), 5)
            lo, hi = np.percentile(var, 4), np.percentile(var, 96)
            norm = np.clip((var - lo) / max(1e-6, hi - lo), 0, 1)
            stops = np.array([0.0, 0.35, 0.68, 1.0])
            colors = np.array([[255, 77, 94], [246, 196, 91], [176, 232, 128], [56, 226, 156]], np.float32)
            lut = np.stack([np.interp(np.linspace(0, 1, 256), stops, colors[:, i]) for i in range(3)], 1).astype(np.uint8)
            heat = lut[(norm * 255).astype(np.uint8)]
            out = (arr.astype(np.float32) * 0.72 + heat.astype(np.float32) * 0.28).astype(np.uint8)
            return Image.fromarray(out)
        except Exception:
            return img

    def _redraw(self):
        if self._closing:
            return
        c = self.canvas
        c.delete("all")
        cw, ch = max(1, c.winfo_width()), max(1, c.winfo_height())
        if self.preview is None:
            if self.current and self.current.is_error:
                c.create_text(cw / 2, ch / 2 - 30, text="无法读取", fill=CORAL, font=self.app.fonts.h1)
                reason = ellipsize(self.current.error_reason(), self.app.font_obj(self.app.fonts.small), cw - 100)
                c.create_text(cw / 2, ch / 2 + 6, text=reason, fill="#d9a2ab", font=self.app.fonts.small)
                c.create_text(cw / 2, ch / 2 + 34, text="可点下方「重试读取」，或用「系统打开」检查原文件",
                              fill=FAINT, font=self.app.fonts.tiny)
            elif self.current:
                c.create_text(cw / 2, ch / 2, text="正在载入大图…", fill=MUTED, font=self.app.fonts.h2)
            return
        if self.current and self.current.is_error:
            c.create_text(cw / 2, ch / 2, text="无法读取", fill=CORAL, font=self.app.fonts.h1)
            return
        # Once the full image is in memory, use it for every zoom level.  Using
        # the 1800px preview at fit and the full file after a wheel step would
        # make a 6000px photo jump from "fit" to ~20% when crossing 1.0x.
        source = self.full if self.full is not None else self.preview
        iw, ih = source.size
        fit = self._fit_scale(iw, ih, cw, ch)
        scale = 1.0 if self.pixel_1to1 else fit * self.zoom
        self._last_scale = scale
        sw, sh = iw * scale, ih * scale
        ox = (cw - sw) / 2 + self.pan[0]
        oy = (ch - sh) / 2 + self.pan[1]

        # Compute the visible source rectangle in image coordinates.  This
        # works for fit, arbitrary zoom and true 1:1, including images smaller
        # than the window.
        sx0 = max(0.0, min(iw, (0 - ox) / max(scale, 1e-6)))
        sx1 = max(0.0, min(iw, (cw - ox) / max(scale, 1e-6)))
        sy0 = max(0.0, min(ih, (0 - oy) / max(scale, 1e-6)))
        sy1 = max(0.0, min(ih, (ch - oy) / max(scale, 1e-6)))
        if sx1 > sx0 and sy1 > sy0:
            crop = source.crop((int(sx0), int(sy0), max(int(sx0) + 1, int(sx1)), max(int(sy0) + 1, int(sy1))))
            dw, dh = max(1, round((sx1 - sx0) * scale)), max(1, round((sy1 - sy0) * scale))
            draw = crop.resize((dw, dh), Image.Resampling.LANCZOS)
            if self.heat:
                draw = self._heat_overlay(draw)
            self._tk_img = ImageTk.PhotoImage(draw)
            c.create_image(ox + sx0 * scale, oy + sy0 * scale, image=self._tk_img, anchor="nw")
        box_ox, box_oy, box_scale = ox, oy, scale

        if self.heat:
            c.create_text(16, ch - 14, text="细节少", fill="#ff8b98", font=self.app.fonts.tiny, anchor="w")
            c.create_text(cw - 16, ch - 14, text="细节多", fill="#8ef0bd", font=self.app.fonts.tiny, anchor="e")
        if self.faces and self.current and self.current.result:
            for i, face in enumerate(self.current.result.get("faces", [])):
                bx, by, bw, bh = face["box"]
                x1, y1 = box_ox + bx * iw * box_scale, box_oy + by * ih * box_scale
                x2, y2 = box_ox + (bx + bw) * iw * box_scale, box_oy + (by + bh) * ih * box_scale
                color = BLUE if face.get("occlusion", {}).get("occluded") else (
                    CORAL if face.get("score", 100) < self.app.threshold - 10 else MINT)
                c.create_rectangle(x1, y1, x2, y2, outline=color, width=1)
                c.create_rectangle(x1, max(0, y1 - 17), x1 + 17, max(17, y1), fill=color, outline="")
                c.create_text(x1 + 8.5, max(8.5, y1 - 8.5), text=str(i + 1), fill="#0c1017", font=self.app.fonts.tiny)
        focus = (self.current.result or {}).get("focus") if self.current else None
        if self.focus and focus and not self.current.is_error:
            fx = box_ox + float(focus.get("x", 0.5)) * iw * box_scale
            fy = box_oy + float(focus.get("y", 0.5)) * ih * box_scale
            c.create_oval(fx - 8, fy - 8, fx + 8, fy + 8, outline="#d9bd87", width=1, dash=(2, 4))
            c.create_oval(fx - 1.5, fy - 1.5, fx + 1.5, fy + 1.5, fill="#d9bd87", outline="")
            label = str(focus.get("label") or ("人脸" if focus.get("source") == "face" else "细节"))
            text = f"估计焦点 · {label}"
            anchor = "w"
            tx, ty = fx + 11, fy
            if focus.get("source") == "face" and self.current.result:
                try:
                    idx = int(str(focus.get("label", "")).split()[-1]) - 1
                    face = self.current.result.get("faces", [])[idx]
                    bx, by, bw, bh = face["box"]
                    tx = box_ox + (bx + bw / 2) * iw * box_scale
                    ty = box_oy + (by + bh) * ih * box_scale + 4
                    anchor = "n"
                    if ty > ch - 16:
                        ty = box_oy + by * ih * box_scale - 4
                        anchor = "s"
                except (ValueError, IndexError, KeyError, TypeError):
                    pass
            if anchor == "w" and tx + self.app.measure(text, self.app.fonts.tiny) > cw - 6:
                anchor, tx = "e", fx - 11
            c.create_text(tx, ty, text=text, fill="#c9ad78", font=self.app.fonts.tiny, anchor=anchor)
        pct = round(scale * 100)
        c.create_text(16, 12, text=f"{pct}%  原图像素" if abs(scale - 1.0) < 0.02 else f"{pct}%",
                      fill=FAINT, font=self.app.fonts.tiny, anchor="nw")

    def _wheel(self, event):
        self.zoom_by(1.15 if event.delta > 0 else 0.87)
        self._redraw()

    def _drag_start(self, event):
        self._drag = (event.x, event.y, self.pan[0], self.pan[1])

    def _drag_move(self, event):
        if self._drag:
            x0, y0, px, py = self._drag
            self.pan = [px + event.x - x0, py + event.y - y0]
            self._redraw()

    def zoom_by(self, factor):
        self.zoom = max(0.1, min(8.0, self.zoom * factor))
        self.pixel_1to1 = False
        self._pending_one_to_one = False
        if self.zoom > 1.01 and self.full is None and self.current and not self.current.is_error:
            self._request(self.current, full=True)
        self._redraw()

    def fit(self):
        self.zoom = 1.0
        self.pan = [0, 0]
        self.pixel_1to1 = False
        self._pending_one_to_one = False
        self._redraw()

    def one_to_one(self):
        if not self.preview or not self.current or self.current.is_error:
            return
        if self.full is None:
            # The 1800px screen preview is not enough for true 1:1; fetch the
            # original (capped in memory) and apply this once it arrives.
            self._pending_one_to_one = True
            self._request(self.current, full=True)
            self._redraw()
            return
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        fit = self._fit_scale(self.full.width, self.full.height, cw, ch)
        self.pixel_1to1 = True
        self.zoom = 1.0 / fit
        self.pan = [0, 0]
        self._redraw()

    def toggle_fit_100(self):
        if self.pixel_1to1 or abs(self._last_scale - 1.0) < 0.02:
            self.fit()
        else:
            self.one_to_one()

    def toggle_heat(self):
        self.heat = not self.heat
        self.heat_btn.set_text(f"热力：{'开' if self.heat else '关'}")
        self.heat_btn.variant = "ghost" if self.heat else "subtle"
        self.heat_btn._draw()
        self.app.settings["preview_heatmap"] = self.heat
        self.app._save_settings()
        self._redraw()

    def toggle_faces(self):
        self.faces = not self.faces
        self.face_btn.set_text(f"人脸：{'开' if self.faces else '关'}")
        self.face_btn.variant = "ghost" if self.faces else "subtle"
        self.face_btn._draw()
        self.app.settings["preview_faces"] = self.faces
        self.app._save_settings()
        self._redraw()

    def toggle_focus(self):
        self.focus = not self.focus
        self.focus_btn.set_text(f"焦点：{'开' if self.focus else '关'}")
        self.focus_btn.variant = "ghost" if self.focus else "subtle"
        self.focus_btn._draw()
        self.app.settings["preview_focus"] = self.focus
        self.app._save_settings()
        self._redraw()

    # ------------------------------------------------------------------ actions
    def go(self, delta):
        if not self.photos:
            return
        self.index = (self.index + delta) % len(self.photos)
        self._show_index()
        self._select_current()

    def go_to(self, index):
        if not self.photos:
            return
        self.index = max(0, min(len(self.photos) - 1, index))
        self._show_index()
        self._select_current()

    def _select_current(self):
        if self.current:
            self.app.selected_key = self.current.key
            self.app.selected_keys = {self.current.key}
            self.app.grid.schedule_render()
            self.app._update_statusbar()

    def delete_current(self):
        if not self.current:
            return
        key = self.current.key
        if not self.app.delete_photos([key], parent=self):
            return
        self.photos = [p for p in self.photos if p.key != key]
        if not self.photos:
            self.close()
            return
        self.index = min(self.index, len(self.photos) - 1)
        self._show_index()

    def retry_current(self):
        if not self.current:
            return
        key = self.current.key
        self.preview = None
        self.full = None
        self.app.retry_photo(key)
        self._update_header()
        self._redraw()
        self.after(250, lambda: self._watch_retry(key, 40))

    def _watch_retry(self, key: str, remaining: int):
        if self._closing or not self.current or self.current.key != key:
            return
        if self.current.result and not self.current.is_error:
            self._show_index()
            return
        if self.current.is_error:
            self._update_header()
            self._redraw()
            return
        if remaining > 0:
            self.after(350, lambda: self._watch_retry(key, remaining - 1))

    def refresh_after_removal(self):
        if self._closing:
            return
        old_key = self.current.key if self.current else ""
        self.photos = [p for p in self.photos if p.key in self.app.by_key]
        if not self.photos:
            self.close()
            return
        keys = [p.key for p in self.photos]
        if old_key in keys:
            self.index = keys.index(old_key)
        else:
            self.index = min(self.index, len(self.photos) - 1)
        self._show_index()

    def external(self):
        if self.current:
            self.app.open_external(self.current.key)

    def close(self):
        if self._closing:
            return
        self._closing = True
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            self.app.on_preview_closed()
        except Exception:
            pass
        try:
            self.destroy()
        except tk.TclError:
            pass


# ---------------------------------------------------------------------------
# duplicate-format manager
# ---------------------------------------------------------------------------
class DuplicateDialog(tk.Toplevel):
    def __init__(self, parent, app: FocusSortApp):
        super().__init__(parent)
        self.app = app
        self.title("同图重复 · 不同格式")
        self.configure(bg=BG)
        self.geometry("820x620")
        self.transient(parent)
        self.grab_set()
        self._build()
        self.bind("<Escape>", lambda _e: self.destroy())

    def _build(self):
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=22, pady=(18, 8))
        tk.Label(head, text="同一照片的不同格式", bg=BG, fg=TEXT, font=self.app.fonts.h1).pack(side="left")
        tk.Label(head, text=f"{len(self.app.dup_groups)} 组 · {self.app.dup_count} 个文件", bg=BG, fg=MUTED, font=self.app.fonts.small).pack(side="left", padx=(12, 0), pady=(6, 0))
        PillButton(head, "关闭", self.destroy, variant="ghost", width=64, height=30, font=self.app.fonts.small).pack(side="right")

        info = tk.Label(
            self,
            text="根据感知哈希 + 画面比例判断“看起来是同一张照片”。不同格式/尺寸会分到一组；不同裁切或明显修图可能不会。\n"
                 "选择一种保留策略后，其余文件会移入 Windows 回收站，可恢复。",
            bg=BG, fg=MUTED, font=self.app.fonts.small, justify="left", wraplength=760,
        )
        info.pack(fill="x", padx=22, pady=(0, 10))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=22)
        text = tk.Text(body, bg=PANEL, fg=TEXT_SOFT, bd=0, highlightthickness=1, highlightbackground=BORDER,
                       font=self.app.fonts.small, wrap="none", padx=14, pady=12)
        scroll = tk.Scrollbar(body, command=text.yview, width=10, bg=PANEL_2, troughcolor=BG, bd=0)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        for gi, group in enumerate(self.app.dup_groups, 1):
            text.insert("end", f"组 {gi} · {len(group)} 个版本\n", "head")
            for p in group:
                r = p.result or {}
                mark = "★" if p.key == self.app.selected_key else " "
                text.insert("end", f" {mark} {p.name:<32} {r.get('width', 0)}×{r.get('height', 0)}  "
                                    f"{human_size(p.size):>9}  {r.get('score', 0):>5.0f} 分\n")
            text.insert("end", "\n")
        text.tag_configure("head", foreground=VIOLET, font=self.app.fonts.small_bold)
        text.configure(state="disabled")

        actions = tk.Frame(self, bg=BG)
        actions.pack(fill="x", padx=22, pady=16)
        tk.Label(actions, text="每组保留：", bg=BG, fg=TEXT_SOFT, font=self.app.fonts.small).pack(side="left")
        for label, mode in (("清晰度最高", "score"), ("分辨率最高", "pixels"), ("文件最大", "size")):
            PillButton(actions, label, lambda m=mode: self._apply(m), variant="ghost", width=110, height=32,
                       font=self.app.fonts.small).pack(side="left", padx=4)

    def _keeper(self, group: list[Photo], mode: str) -> Photo:
        if mode == "pixels":
            return max(group, key=lambda p: (p.result or {}).get("width", 0) * (p.result or {}).get("height", 0))
        if mode == "size":
            return max(group, key=lambda p: p.size)
        return max(group, key=lambda p: (p.result or {}).get("score", 0))

    def _apply(self, mode: str):
        delete: list[Photo] = []
        for group in self.app.dup_groups:
            keeper = self._keeper(group, mode)
            delete.extend(p for p in group if p is not keeper)
        if not delete:
            return
        labels = {"score": "清晰度最高", "pixels": "分辨率最高", "size": "文件最大"}
        if not messagebox.askyesno(
            "同图去重",
            f"每组保留{labels[mode]}的版本，将 {len(delete)} 个其余文件移入回收站？\n\n"
            "这是可恢复操作，不会永久删除。",
            parent=self,
        ):
            return
        if self.app.delete_photos([p.key for p in delete], confirm=False, parent=self):
            self.destroy()


# ---------------------------------------------------------------------------
# settings / maintenance window
# ---------------------------------------------------------------------------
class SettingsWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, app: FocusSortApp):
        super().__init__(parent)
        self.app = app
        self.title("设置与维护 · FocusSort")
        self.configure(bg=BG)
        self.geometry("650x560")
        self.minsize(600, 520)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", lambda _e: self.close())
        self._build()
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _section(self, title: str) -> tk.Frame:
        box = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        box.pack(fill="x", padx=22, pady=(10, 0))
        tk.Label(box, text=title, bg=PANEL, fg=TEXT, font=self.app.fonts.h3, anchor="w").pack(
            fill="x", padx=14, pady=(11, 5)
        )
        return box

    def _build(self) -> None:
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=22, pady=(18, 2))
        tk.Label(head, text="设置与维护", bg=BG, fg=TEXT, font=self.app.fonts.h1).pack(side="left")
        PillButton(head, "关闭", self.close, variant="ghost", width=64, height=30,
                   font=self.app.fonts.small).pack(side="right")
        tk.Label(self, text="所有维护操作都不会删除、移动或修改原照片。", bg=BG, fg=MUTED,
                 font=self.app.fonts.small, anchor="w").pack(fill="x", padx=22, pady=(4, 0))

        session = self._section("上次会话")
        row = tk.Frame(session, bg=PANEL)
        row.pack(fill="x", padx=14, pady=(0, 4))
        txt = tk.Frame(row, bg=PANEL)
        txt.pack(side="left", fill="x", expand=True)
        tk.Label(txt, text="启用上次会话恢复", bg=PANEL, fg=TEXT_SOFT, font=self.app.fonts.body_bold, anchor="w").pack(fill="x")
        tk.Label(
            txt,
            text=f"启动时检查上次照片是否仍存在、可读且格式受支持；失效记录自动清理。最多恢复 {MAX_RESTORE_FILES} 张。",
            bg=PANEL, fg=MUTED, font=self.app.fonts.tiny, anchor="w", justify="left", wraplength=460,
        ).pack(fill="x", pady=(2, 0))
        ToggleSwitch(
            row,
            bool(self.app.settings.get("restore_session", True)),
            self._set_restore,
        ).pack(side="right", padx=(12, 0))
        self.restore_note = tk.Label(
            session,
            text="当前已启用" if self.app.settings.get("restore_session", True) else "当前已关闭",
            bg=PANEL, fg=FAINT, font=self.app.fonts.tiny, anchor="w",
        )
        self.restore_note.pack(fill="x", padx=14, pady=(2, 10))

        current = self._section("当前列表")
        actions = tk.Frame(current, bg=PANEL)
        actions.pack(fill="x", padx=14, pady=(0, 12))
        PillButton(actions, "清理无效记录", self._cleanup, variant="subtle", width=112, height=30,
                   font=self.app.fonts.small).pack(side="left", padx=(0, 6))
        PillButton(actions, "清空当前列表", self._clear_list, variant="subtle", width=112, height=30,
                   font=self.app.fonts.small).pack(side="left")
        tk.Label(actions, text="不删除文件", bg=PANEL, fg=FAINT, font=self.app.fonts.tiny).pack(side="left", padx=10)

        cache = self._section("本地缓存")
        self.stats = tk.Label(cache, text=self.app.cache_summary_text(), bg=PANEL, fg=MUTED,
                              font=self.app.fonts.tiny, anchor="w", justify="left", wraplength=570)
        self.stats.pack(fill="x", padx=14, pady=(0, 8))
        days_row = tk.Frame(cache, bg=PANEL)
        days_row.pack(fill="x", padx=14, pady=(0, 6))
        tk.Label(days_row, text="按时间清理：", bg=PANEL, fg=TEXT_SOFT, font=self.app.fonts.small).pack(side="left")
        self.days_var = tk.StringVar(value=str(self.app.settings.get("cache_days", DEFAULT_CACHE_DAYS)))
        spin = tk.Spinbox(
            days_row, from_=1, to=365, width=5, textvariable=self.days_var,
            bg="#1a1f2a", fg=TEXT, buttonbackground=PANEL_2, insertbackground=TEXT,
            relief="flat", highlightthickness=1, highlightbackground=BORDER, font=self.app.fonts.small,
        )
        spin.pack(side="left", padx=(4, 4))
        tk.Label(days_row, text="天未被使用的缓存", bg=PANEL, fg=MUTED, font=self.app.fonts.tiny).pack(side="left")
        PillButton(days_row, "清理未使用", self._purge_old, variant="subtle", width=92, height=28,
                   font=self.app.fonts.small).pack(side="right")
        actions2 = tk.Frame(cache, bg=PANEL)
        actions2.pack(fill="x", padx=14, pady=(0, 12))
        PillButton(actions2, "清理失效缓存", self._purge_missing, variant="subtle", width=112, height=30,
                   font=self.app.fonts.small).pack(side="left", padx=(0, 6))
        PillButton(actions2, "清空全部缓存", self._purge_all, variant="subtle", width=112, height=30,
                   font=self.app.fonts.small).pack(side="left", padx=(0, 6))
        PillButton(actions2, "打开缓存文件夹", self._open_cache, variant="subtle", width=124, height=30,
                   font=self.app.fonts.small).pack(side="right")
        tk.Label(
            cache,
            text="缓存仅用于加快再次导入；清空不会影响照片。",
            bg=PANEL, fg=FAINT, font=self.app.fonts.tiny, anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 10))

    def _set_restore(self, value: bool) -> None:
        self.app.settings["restore_session"] = bool(value)
        self.app._save_settings()
        self.restore_note.configure(text="当前已启用" if value else "当前已关闭")

    def _cleanup(self) -> None:
        self.app.cleanup_invalid_records()
        self.refresh_stats()

    def _clear_list(self) -> None:
        self.app.clear_photo_list()
        self.refresh_stats()

    def _days(self) -> int:
        try:
            return max(1, min(3650, int(float(self.days_var.get()))))
        except (TypeError, ValueError):
            return DEFAULT_CACHE_DAYS

    def _purge_missing(self) -> None:
        self.app.purge_cache("missing")
        self.refresh_stats()

    def _purge_old(self) -> None:
        self.app.settings["cache_days"] = self._days()
        self.app._save_settings()
        self.app.purge_cache("old", self._days())
        self.refresh_stats()

    def _purge_all(self) -> None:
        self.app.purge_cache("all")
        self.refresh_stats()

    def _open_cache(self) -> None:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            os.startfile(str(CACHE_DIR))
        except Exception as exc:
            self.app.toast.show(f"无法打开缓存文件夹：{exc}")

    def refresh_stats(self) -> None:
        try:
            self.stats.configure(text=self.app.cache_summary_text())
        except tk.TclError:
            pass

    def close(self) -> None:
        self.app.settings_window = None
        try:
            self.destroy()
        except tk.TclError:
            pass


# ---------------------------------------------------------------------------
# help window
# ---------------------------------------------------------------------------
class HelpWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, app: FocusSortApp):
        super().__init__(parent)
        self.app = app
        self.title("使用说明 · FocusSort")
        self.configure(bg=BG)
        self.geometry("760x650")
        self.minsize(640, 520)
        self.transient(parent)
        self.grab_set()
        self._build()
        self.bind("<Escape>", lambda _e: self.destroy())
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _build(self):
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=24, pady=(20, 8))
        tk.Label(head, text="怎么用 FocusSort", bg=BG, fg=TEXT, font=self.app.fonts.h1).pack(side="left")
        PillButton(head, "关闭", self.destroy, variant="ghost", width=68, height=30, font=self.app.fonts.small).pack(side="right")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=(4, 18))
        text = tk.Text(body, bg=PANEL, fg=TEXT_SOFT, bd=0, highlightthickness=1, highlightbackground=BORDER,
                       font=self.app.fonts.body, wrap="char", padx=20, pady=16, spacing1=2, spacing3=6,
                       insertbackground=TEXT, cursor="arrow")
        scroll = tk.Scrollbar(body, command=text.yview, bg=PANEL_2, troughcolor=BG, bd=0, width=10)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        text.tag_configure("h2", foreground=ACCENT, font=self.app.fonts.h3, spacing1=14, spacing3=4)
        text.tag_configure("muted", foreground=MUTED, font=self.app.fonts.small)
        text.tag_configure("key", foreground=TEXT, font=self.app.fonts.small_bold)
        text.tag_configure("good", foreground=MINT)
        text.tag_configure("warn", foreground=AMBER)
        text.tag_configure("bad", foreground=CORAL)

        def add(s: str, tag: str | None = None):
            text.insert("end", s, tag or ())

        add("三步开始\n", "h2")
        add("1. 把照片或整个文件夹拖进窗口；也可以点右上角「照片 / 文件夹」。\n")
        add("2. 等待缩略图上的分数出现。分析在本机进行，不会上传。\n")
        add("3. 用左侧筛选和判定线快速圈出可疑照片，点击缩略图在右侧复核。\n\n")
        add("分数怎么看\n", "h2")
        add("分数 0–100，比较的是「画面里还剩多少可分辨的细节」。有清晰人脸时，分数会优先参考人脸；")
        add("没有可靠人脸时，参考整体纹理、边缘与频谱。\n")
        add("• ", None); add("清晰", "good"); add("：达到判定线。\n")
        add("• ", None); add("待复核", "warn"); add("：接近判定线，或画面细节太少。建议看 100% 细节。\n")
        add("• ", None); add("模糊", "bad"); add("：明显低于判定线。\n")
        add("• ", None); add("细节少", "muted"); add("：大片天空、纯色墙面等本身缺细节，分数参考性低，不一定是失焦。\n\n")
        add("面部与遮挡\n", "h2")
        add("会同时尝试 BlazeFace 与 Haar 检测人脸，再把脸部按眼睛/嘴巴位置摆正，检查肤色、眼部暗区和纹理。")
        add("能识别口罩、墨镜/深色眼镜、衣领等造成的明显遮挡；肤色接近的手掌、头发遮挡可能漏判。")
        add("结果只是启发式估计，重要照片请用右侧 100% 细节确认。\n\n")
        add("预览操作\n", "h2")
        add("• 单击预览任意位置 → 下方 100% 放大镜对准该处。\n")
        add("• 双击缩略图 / 空格 / 回车 → 打开大图预览页（不会调用系统看图应用）。\n")
        add("• 大图预览里 ← → 连续翻阅同一筛选结果，滚轮缩放，双击在 1:1/适应间切换，Esc 返回图库。\n")
        add("• 右键缩略图可选择「用系统应用打开」。\n")
        add("• 「细节热力」：绿色=细节多，红色=细节少；反映细节密度，不等于绝对失焦。\n")
        add("• 「焦点」默认关闭。打开后只是细小半透明感圆环 + 一行小字，不是大面积准星。\n")
        add("• 估计焦点优先取最清晰人脸中心，没有可靠人脸时取最锐细节区域；这是单张景深推断，不是相机 AF 元数据。\n")
        add("• 「人脸框」默认关闭：蓝框=疑似遮挡，红框=人脸偏虚，绿框=正常。\n")
        add("• 「自动定线」只给出建议，点击建议文字才会应用，不会偷偷修改当前判定线。\n\n")
        add("快捷键\n", "h2")
        shortcuts = [
            ("Ctrl+O", "添加照片"),
            ("Ctrl+Shift+O", "添加文件夹"),
            ("← → ↑ ↓", "连续翻阅当前筛选结果"),
            ("Ctrl / Shift + 单击", "多选照片"),
            ("Ctrl+A", "全选当前筛选"),
            ("空格 / 回车", "打开大图预览页"),
            ("1 – 7", "全部 / 清晰 / 待复核 / 模糊 / 异常 / 有遮挡 / 同图重复"),
            ("H", "切换细节热力图"),
            ("F", "切换人脸框"),
            ("G", "切换估计焦点细环"),
            ("+ / -", "微调判定线"),
            ("F5", "重新分析当前照片"),
            ("Delete", "批量移入回收站（可恢复）"),
            ("Ctrl+E", "导出当前筛选 CSV"),
            ("Esc", "关闭预览/弹窗，或清除主窗口选择"),
        ]
        for key, desc in shortcuts:
            add(f"{key:<14}", "key")
            add(f"{desc}\n", "muted")
        add("\n同图与批量删除\n", "h2")
        add("不同格式/尺寸的同一张照片会显示紫色的「同图×N」，左侧有「同图重复」筛选。")
        add("点击右侧紫色提示，可按清晰度、分辨率或文件大小保留一张，其余移入回收站。\n")
        add("多选照片后底部会出现批量条；所有删除都进入 Windows 回收站，程序不会永久删除。\n")
        add("\n异常与 HEIC\n", "h2")
        add("读取失败的卡片会显示原因，并提供重试、从列表移除、打开所在文件夹；左侧「异常」可集中处理。")
        add("HEIC/HEIF 需要可选的 pillow-heif 解码器；没有安装时程序会明确提示，而不是笼统报错。\n")
        add("\n会话与缓存\n", "h2")
        add(f"分析结果和缩略图缓存在：\n{DATA_DIR}\n", "muted")
        add("右上角齿轮可设置「启用上次会话恢复」、清理无效记录、清空当前列表、按时间清理缓存。")
        add("启动时会逐条校验上次照片，自动移除失效记录并提示数量；最多恢复 300 张。")
        add("清理缓存不会影响原照片。\n")
        text.configure(state="disabled")


# ---------------------------------------------------------------------------
def main():
    app = FocusSortApp()
    paths: list[Path] = []
    select = ""
    want_heat = False
    want_help = False
    want_preview = False
    smoke_test = False
    explicit_paths = False
    for arg in sys.argv[1:]:
        if arg.startswith("--select="):
            select = arg.split("=", 1)[1]
        elif arg == "--heat":
            want_heat = True
        elif arg == "--help":
            want_help = True
        elif arg == "--preview":
            want_preview = True
        elif arg == "--smoke-test":
            # Used by the automated launcher check; normal users never need it.
            smoke_test = True
        else:
            paths.append(Path(arg))
            explicit_paths = True

    restored_invalid = 0
    restored_limited = 0
    if not paths and not want_help and app.settings.get("restore_session", True):
        valid, restored_invalid, restored_limited = app.restore_last_session()
        paths = valid
        if valid:
            want_preview = True
        if restored_invalid:
            app.root.after(650, lambda: app.toast.show(
                f"已清理 {restored_invalid} 个失效文件（已删除/移动/无法读取），未恢复这些记录", 3800
            ))
        if restored_limited:
            app.root.after(1200, lambda: app.toast.show(
                f"上次会话超过 {MAX_RESTORE_FILES} 张上限，已只恢复前 {MAX_RESTORE_FILES} 张", 3600
            ))

    if paths:
        initial = list(paths)
        app.root.after(250, lambda: app._add_paths(initial, from_dialog=True))
        if not explicit_paths:
            want_preview = True
        elif not want_preview:
            # Explicitly opened paths should also go straight to the viewer,
            # matching the old command-line behavior.
            want_preview = True
    if select:
        def pick():
            app.set_filter("all")
            for p in app.photos:
                if select.lower() in p.name.lower():
                    app.select(p.key)
                    if want_heat:
                        app.inspector._set_heat("heat")
                    return
        app.root.after(1500, pick)
    if want_preview and paths:
        app.root.after(2100, app.open_preview)
    if want_help:
        app.root.after(500, app.show_help)
    if smoke_test:
        app.root.after(4500, app.on_close)
    app.root.mainloop()


if __name__ == "__main__":
    main()
