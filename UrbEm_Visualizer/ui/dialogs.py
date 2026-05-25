from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog


def _run_dialog(fn):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return fn()
    finally:
        root.destroy()


def pick_yaml_open() -> str | None:
    def go():
        return filedialog.askopenfilename(
            title="Load configuration (YAML)",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )

    path = _run_dialog(go)
    return path if path else None


def pick_yaml_save() -> str | None:
    def go():
        return filedialog.asksaveasfilename(
            title="Save configuration (YAML)",
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )

    path = _run_dialog(go)
    return path if path else None


def pick_folder(title: str) -> str | None:
    def go():
        return filedialog.askdirectory(title=title)

    path = _run_dialog(go)
    return path if path else None


def pick_file(title: str, patterns: list[tuple[str, str]] | None = None) -> str | None:
    types = patterns or [("All files", "*.*")]

    def go():
        return filedialog.askopenfilename(title=title, filetypes=types)

    path = _run_dialog(go)
    return path if path else None
