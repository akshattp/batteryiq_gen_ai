"""scripts/make_colab_pack.py - Create lightweight colab_pack.zip for Google Colab handoff.

Packages ONLY what Google Colab requires:
- scripts/train_bert.py, scripts/train_lora.py
- data/incidents/*.csv, data/lora/*.jsonl
- requirements-colab.txt
- notebooks/colab_train_bert.ipynb, notebooks/colab_train_lora.ipynb
"""

from __future__ import annotations

import os
from pathlib import Path
import zipfile

FILES_TO_PACK = [
    # Training scripts
    "scripts/train_bert.py",
    "scripts/train_lora.py",
    # Datasets
    "data/incidents/incidents.csv",
    "data/incidents/incident_reports.csv",
    "data/lora/train.jsonl",
    "data/lora/val.jsonl",
    # Requirements
    "requirements-colab.txt",
    # Notebooks
    "notebooks/colab_train_bert.ipynb",
    "notebooks/colab_train_lora.ipynb",
]

OUTPUT_ZIP = Path("colab_pack.zip")


def create_colab_pack() -> Path:
    """Create colab_pack.zip containing only Colab handoff dependencies."""
    print("Building Colab handoff pack (colab_pack.zip)...")
    packed_files = []

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_str in FILES_TO_PACK:
            p = Path(file_str)
            if p.exists():
                zf.write(p, arcname=p.as_posix())
                packed_files.append(p.as_posix())
                print(f"  + Added: {p.as_posix()} ({p.stat().st_size / 1024:.1f} KB)")
            else:
                print(f"  ! Warning: {file_str} not found, skipping.")

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\n[COLAB PACK READY] Total files: {len(packed_files)}, Size: {size_mb:.2f} MB")
    print(f"Archive written to: {OUTPUT_ZIP.resolve()}")
    return OUTPUT_ZIP


if __name__ == "__main__":
    create_colab_pack()
