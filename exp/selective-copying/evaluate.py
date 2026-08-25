"""
Selective Copying evaluation — thin launcher over exp/copying/evaluate.py
(same sys.modules["task"] swap as train.py; see there for details).

Usage: same flags as exp/copying/evaluate.py.
"""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task  # noqa: F401  (binds selective task as sys.modules["task"])

assert sys.modules["task"].TASK_NAME == "selective-copying"
runpy.run_path(str(HERE.parent / "copying" / "evaluate.py"), run_name="__main__")
