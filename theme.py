"""Visual language and small custom widgets for FocusSort."""
from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from typing import Callable

# ------------------------------------------------------------------ palette
BG = "#0d0f14"
PANEL = "#12151c"
PANEL_2 = "#161a23"
CARD = "#181c26"
CARD_HOVER = "#1e2330"
CARD_PRESSED = "#222938"
BORDER = "#252b38"
BORDER_LIGHT = "#343c4e"
TEXT = "#edf0f6"
TEXT_SOFT = "#c8cedb"
MUTED = "#8a93a6"
FAINT = "#5f687a"
ACCENT = "#f2b35c"
ACCENT_HOVER = "#ffc978"
ACCENT_DARK = "#3b2a16"
MINT = "#5ee0a0"
AMBER = "#f6c45b"
CORAL = "#ff7185"
BLUE = "#79a8ff"
VIOLET = "#b99cff"
WHITE = "#ffffff"
BLACK = "#05070a"

STATUS_COLORS = {
    "pending": MUTED,
    "cancelled": FAINT,
    "error": CORAL,
    "sharp": MINT,
    "review": AMBER,
    "blur": CORAL,
    "uncertain": BLUE,
}

STATUS_LABELS = {
    "pending": "分析中",
    "cancelled": "已取消",
    "error": "无法读取",
    "sharp": "清晰",
    "review": "待复核",
    "blur": "模糊",
    "uncertain": "细节少",
}


class Fonts:
    def __init__(self, root: tk.Misc):
        families = set(tkfont.families(root))
        for candidate in ("Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Segoe UI"):
            if candidate in families:
                self.family = candidate
                break
        else:
            self.family = "TkDefaultFont"
        self.display = (self.family, 23, "bold")
        self.h1 = (self.family, 18, "bold")
        self.h2 = (self.family, 13, "bold")
        self.h3 = (self.family, 11, "bold")
        self.body = (self.family, 10)
        self.body_bold = (self.family, 10, "bold")
        self.small = (self.family, 9)
        self.small_bold = (self.family, 9, "bold")
        self.tiny = (self.family, 8)
        self.number = (self.family, 34, "bold")
        self.mono = ("Cascadia Mono", 9) if "Cascadia Mono" in families else ("Consolas", 9)


# ------------------------------------------------------------------ drawing
def round_rect(canvas: tk.Canvas, x1: float, y1: float, x2: float, y2: float, radius: float, **kwargs) -> int:
    radius = max(0.0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=24, **kwargs)


def ellipsize(text: str, font: tkfont.Font, max_width: int) -> str:
    if max_width <= 0:
        return ""
    if font.measure(text) <= max_width:
        return text
    ell = "…"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if font.measure(text[:mid] + ell) <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + ell if lo else ell


def natural_key(text: str) -> list:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def human_size(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} GB"


# ------------------------------------------------------------------ tooltip
class ToolTip:
    def __init__(self, widget: tk.Misc, text: str = "", delay: int = 450):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._job = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def set_text(self, text: str) -> None:
        self.text = text

    def _schedule(self, _event=None) -> None:
        self._cancel()
        if self.text:
            self._job = self.widget.after(self.delay, self._show)

    def _cancel(self) -> None:
        if self._job:
            try:
                self.widget.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None

    def _show(self) -> None:
        if self._tip or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        except tk.TclError:
            return
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.configure(bg=BORDER_LIGHT)
        label = tk.Label(
            tip,
            text=self.text,
            bg="#202634",
            fg=TEXT_SOFT,
            font=Fonts(self.widget).small,
            padx=9,
            pady=5,
            justify="left",
        )
        label.pack(padx=1, pady=1)
        tip.wm_geometry(f"+{x}+{y}")
        self._tip = tip

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


# ------------------------------------------------------------------ buttons
class PillButton(tk.Canvas):
    """A compact, custom-drawn button with hover states."""

    VARIANTS = {
        "primary": {"bg": ACCENT, "fg": "#1a1309", "hover": ACCENT_HOVER, "border": ""},
        "ghost": {"bg": PANEL_2, "fg": TEXT_SOFT, "hover": CARD_PRESSED, "border": BORDER},
        "subtle": {"bg": "", "fg": MUTED, "hover": CARD_PRESSED, "border": ""},
        "danger": {"bg": "#321b22", "fg": "#ff9aa8", "hover": "#43232d", "border": "#512a36"},
    }

    def __init__(
        self,
        master,
        text: str,
        command: Callable[[], None],
        *,
        variant: str = "ghost",
        width: int | None = None,
        height: int = 34,
        font: tuple | None = None,
        icon: str = "",
        radius: int = 10,
        padx: int = 14,
        tooltip: str = "",
    ):
        self._font_obj = tkfont.Font(master, font=font or Fonts(master).body_bold)
        label = f"{icon}  {text}".strip() if icon else text
        w = width or (self._font_obj.measure(label) + padx * 2)
        super().__init__(
            master,
            width=w,
            height=height,
            bg=master.cget("bg") if "bg" in master.keys() else PANEL,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.text = label
        self.command = command
        self.variant = variant
        self.radius = radius
        self._enabled = True
        self._hover = False
        self._pressed = False
        self._btn_w = w
        self._btn_h = height
        self._font = font or Fonts(master).body_bold
        self._draw()
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        if tooltip:
            ToolTip(self, tooltip)

    def _draw(self) -> None:
        self.delete("all")
        v = self.VARIANTS[self.variant]
        if not self._enabled:
            bg, fg = PANEL_2, FAINT
        else:
            bg = v["hover"] if (self._hover or self._pressed) else (v["bg"] or self.master.cget("bg"))
            fg = v["fg"]
        round_rect(
            self,
            1,
            1,
            self._btn_w - 1,
            self._btn_h - 1,
            self.radius,
            fill=bg,
            outline=v["border"] if self._enabled else BORDER,
            width=1,
        )
        self.create_text(
            self._btn_w / 2,
            self._btn_h / 2,
            text=self.text,
            fill=fg,
            font=self._font,
            anchor="center",
        )

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        self._draw()

    def set_text(self, text: str) -> None:
        self.text = text
        self._draw()

    def _on_enter(self, _e=None):
        self._hover = True
        self._draw()

    def _on_leave(self, _e=None):
        self._hover = False
        self._pressed = False
        self._draw()

    def _on_press(self, _e=None):
        if self._enabled:
            self._pressed = True
            self._draw()

    def _on_release(self, _e=None):
        if not self._enabled:
            return
        was = self._pressed
        self._pressed = False
        self._draw()
        if was:
            self.command()


class IconButton(tk.Canvas):
    def __init__(
        self,
        master,
        glyph: str,
        command: Callable[[], None],
        *,
        size: int = 34,
        tooltip: str = "",
        fg: str = TEXT_SOFT,
        hover_bg: str = CARD_PRESSED,
        font: tuple | None = None,
        bg: str | None = None,
    ):
        self._bg = bg or master.cget("bg")
        super().__init__(master, width=size, height=size, bg=self._bg, highlightthickness=0, bd=0, cursor="hand2")
        self.glyph = glyph
        self.command = command
        self.fg = fg
        self.hover_bg = hover_bg
        self.size = size
        self.font = font or (Fonts(master).family, 13, "bold")
        self._hover = False
        self._draw()
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<Button-1>", lambda _e: self.command())
        if tooltip:
            ToolTip(self, tooltip)

    def _draw(self):
        self.delete("all")
        if self._hover:
            round_rect(self, 2, 2, self.size - 2, self.size - 2, 9, fill=self.hover_bg, outline="")
        self.create_text(self.size / 2, self.size / 2 + 1, text=self.glyph, fill=self.fg, font=self.font, anchor="center")

    def _enter(self, _e=None):
        self._hover = True
        self._draw()

    def _leave(self, _e=None):
        self._hover = False
        self._draw()


class ToggleSwitch(tk.Canvas):
    def __init__(self, master, value: bool, command: Callable[[bool], None], *, width: int = 34, height: int = 20):
        bg = master.cget("bg")
        super().__init__(master, width=width, height=height, bg=bg, highlightthickness=0, bd=0, cursor="hand2")
        self.value = bool(value)
        self.command = command
        self.w = width
        self.h = height
        self._draw()
        self.bind("<Button-1>", self._toggle)

    def _draw(self):
        self.delete("all")
        fill = ACCENT if self.value else "#2b3240"
        round_rect(self, 1, 1, self.w - 1, self.h - 1, self.h / 2, fill=fill, outline="")
        cx = self.w - self.h / 2 if self.value else self.h / 2
        r = self.h / 2 - 3
        self.create_oval(cx - r, self.h / 2 - r, cx + r, self.h / 2 + r, fill=WHITE if self.value else "#8a93a6", outline="")

    def _toggle(self, _e=None):
        self.value = not self.value
        self._draw()
        self.command(self.value)


class ThresholdSlider(tk.Canvas):
    """A custom 0..100 slider that matches the dark UI."""

    def __init__(self, master, value: float, command: Callable[[float], None], *, width: int = 178, height: int = 28):
        super().__init__(master, width=width, height=height, bg=master.cget("bg"), highlightthickness=0, bd=0, cursor="hand2")
        self.w = width
        self.h = height
        self.value = float(value)
        self.command = command
        self._dragging = False
        self._draw()
        self.bind("<Button-1>", self._start)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", self._end)
        self.bind("<Double-Button-1>", self._reset)

    def _x_for(self, value: float) -> float:
        return 9 + (self.w - 18) * (value / 100.0)

    def _draw(self):
        self.delete("all")
        x1, x2 = 9, self.w - 9
        cy = self.h / 2
        round_rect(self, x1, cy - 3, x2, cy + 3, 3, fill="#2a3140", outline="")
        kx = self._x_for(self.value)
        round_rect(self, x1, cy - 3, max(x1 + 6, kx), cy + 3, 3, fill=ACCENT, outline="")
        self.create_oval(kx - 7, cy - 7, kx + 7, cy + 7, fill="#f7f8fb", outline="")
        self.create_oval(kx - 3.5, cy - 3.5, kx + 3.5, cy + 3.5, fill=ACCENT, outline="")

    def _set_from_x(self, x: float):
        value = (x - 9) / max(1, (self.w - 18)) * 100.0
        value = max(0.0, min(100.0, value))
        self.value = value
        self._draw()
        self.command(value)

    def _start(self, event):
        self._dragging = True
        self._set_from_x(event.x)

    def _move(self, event):
        if self._dragging:
            self._set_from_x(event.x)

    def _end(self, _event):
        self._dragging = False

    def _reset(self, _event):
        self.value = 60.0
        self._draw()
        self.command(self.value)

    def set_value(self, value: float, notify: bool = False):
        self.value = max(0.0, min(100.0, float(value)))
        self._draw()
        if notify:
            self.command(self.value)


class Segmented(tk.Canvas):
    """Small segmented control for preview modes."""

    def __init__(self, master, options: list[tuple[str, str]], value: str, command: Callable[[str], None], height: int = 30):
        self.font_obj = tkfont.Font(master, font=Fonts(master).small_bold)
        widths = [max(42, self.font_obj.measure(label) + 18) for _, label in options]
        super().__init__(master, width=sum(widths), height=height, bg=master.cget("bg"), highlightthickness=0, bd=0)
        self.options = options
        self.value = value
        self.command = command
        self.widths = widths
        self.h = height
        self._draw()
        self.bind("<Button-1>", self._click)

    def _draw(self):
        self.delete("all")
        round_rect(self, 0, 0, sum(self.widths), self.h, 8, fill=PANEL_2, outline=BORDER)
        x = 0
        for (key, label), w in zip(self.options, self.widths):
            if key == self.value:
                round_rect(self, x + 2, 2, x + w - 2, self.h - 2, 6, fill="#2b3242", outline="")
            self.create_text(
                x + w / 2,
                self.h / 2,
                text=label,
                fill=ACCENT if key == self.value else MUTED,
                font=self.font_obj,
            )
            x += w

    def _click(self, event):
        x = 0
        for (key, _), w in zip(self.options, self.widths):
            if x <= event.x < x + w:
                self.value = key
                self._draw()
                self.command(key)
                return
            x += w

    def set_value(self, value: str):
        self.value = value
        self._draw()
