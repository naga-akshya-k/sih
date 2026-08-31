import sys
from pathlib import Path

# Automatically ensure colonpath_ai is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
