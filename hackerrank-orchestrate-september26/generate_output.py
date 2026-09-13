#!/usr/bin/env python3
"""CLI runner to generate output.csv deterministically from repository root.

Usage:
    python generate_output.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.output import main

if __name__ == "__main__":
    main()
