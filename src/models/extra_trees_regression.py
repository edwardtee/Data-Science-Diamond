import sys
from pathlib import Path

# Navigate up 2 levels from testing.py to find the project root directory
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.preprocessing.preprocessing import load_split

X_train, X_test, y_train, y_test = load_split()
