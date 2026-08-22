"""Compatibility wrapper for the Cosmos Transfer2.5 int4 generation CLI.

The maintained generation entrypoint now lives in:
    /home/yusuke/gitrepos/ad-data-pipeline/pipelines/cosmos_transfer25/

The model-runtime loader lives in:
    /home/yusuke/gitrepos/onecompression-runtime/onecomp_runtime/models/cosmos_transfer25/
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = (
        Path.home()
        / "gitrepos/ad-data-pipeline/pipelines/cosmos_transfer25/"
        / "generate_official_multibranch_int4.py"
    )
    runpy.run_path(str(target), run_name="__main__")
