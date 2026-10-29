"""FocusSort bootstrap.

Started by the .bat launcher with pythonw.exe (no console window).  It checks
the local runtime, exposes the bundled ``vendor`` packages, then hands control
to app.main().  Any startup failure is written to error.log and shown as a
short dialog instead of silently disappearing.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "vendor"))
sys.path.insert(0, str(BASE))
os.chdir(BASE)


def fail(title: str, message: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        print(f"{title}\n{message}", file=sys.stderr)
    raise SystemExit(1)


if sys.version_info < (3, 10):
    fail(
        "FocusSort 需要 Python 3.10+",
        f"当前版本：{sys.version.split()[0]}\n请安装 Python 3.10 或更新版本。",
    )

try:
    import numpy  # noqa: F401
    from PIL import Image  # noqa: F401
    import cv2  # noqa: F401
except Exception as exc:  # noqa: BLE001
    fail(
        "FocusSort 缺少运行库",
        "无法加载图像处理库：\n"
        f"{exc}\n\n"
        "请在 FocusSort 文件夹中运行：\n"
        "  python -m pip install -r requirements.txt\n\n"
        "如果已经安装，请确认 Python 版本是 3.10 或更新版本，并且 vendor 文件夹完整。",
    )


def _log_startup_error(text: str) -> None:
    candidates = [Path(os.getenv("LOCALAPPDATA") or Path.home()) / "FocusSort", BASE / ".data"]
    for folder in candidates:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "error.log").write_text(text, "utf-8")
            return
        except OSError:
            continue


if __name__ == "__main__":
    try:
        import app

        app.main()
    except SystemExit:
        raise
    except Exception:
        text = traceback.format_exc()
        _log_startup_error(text)
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "FocusSort 启动失败",
                "程序遇到一个错误，未能启动。\n\n"
                f"{text[-1200:]}\n\n"
                "详细信息已写入 error.log。",
            )
            root.destroy()
        except Exception:
            print(text, file=sys.stderr)
        raise SystemExit(1)
