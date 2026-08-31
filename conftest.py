import sys
from pathlib import Path

# Automatically ensure colonpath_ai and root are in sys.path for test discovery
CURRENT_DIR = Path(__file__).resolve().parent
COLONPATH_AI_DIR = CURRENT_DIR / "colonpath_ai" if (CURRENT_DIR / "colonpath_ai").exists() else CURRENT_DIR

if str(COLONPATH_AI_DIR) not in sys.path:
    sys.path.insert(0, str(COLONPATH_AI_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
