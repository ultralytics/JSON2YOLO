# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
from pathlib import Path

# Top-level converter scripts live in the repo root, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
