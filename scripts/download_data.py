"""scripts/download_data.py - Data download and discovery verification script."""

from __future__ import annotations

import sys
from pathlib import Path
from guardian.data.ingest import find_mat_files


def main() -> None:
    print("[download_data] Checking for NASA PCoE battery data in data/raw/ ...")
    files = find_mat_files("data/raw")
    if not files:
        print("[download_data] No .mat files found in data/raw/.")
        print("Please place B0005.mat, B0006.mat, B0007.mat, B0018.mat in data/raw/.")
        sys.exit(1)

    print(f"[download_data] Found {len(files)} .mat files:")
    for f in files:
        print(f"  - {f.name} ({f.stat().st_size / (1024*1024):.1f} MB) at {f}")
    print("[download_data] Dataset verified successfully.")


if __name__ == "__main__":
    main()
