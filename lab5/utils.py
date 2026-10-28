"""Helpers: logging to file, CSV, section headers."""
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


class Tee:
    """Writes console output to a log file as well (appends, with a run timestamp)."""

    def __init__(self, path):
        self.stdout = sys.stdout
        self.file = open(path, "a", encoding="utf-8", buffering=1)  # line-buffered
        self.file.write(f"\n\n=== Run started {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def __getattr__(self, name):  
        if name in ("stdout", "file"):
            raise AttributeError(name)
        return getattr(self.stdout, name)


def header(title):
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4f}"
    return str(v)


def write_csv(path, cols, rows):
    lines = [",".join(cols)] + [",".join(_fmt(v) for v in r) for r in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
