"""
Selective Copying training — thin launcher.

Reuses exp/copying/train.py wholesale; the only difference is the task
module. Importing our local `task` FIRST binds it into sys.modules, so when
copying/train.py later does `from task import ...` it resolves to the
selective task (module cache wins over sys.path). Artifacts go to
$DATA_DIR/exp/selective-copying/ via task.TASK_NAME.

Usage: same flags as exp/copying/train.py, e.g.
    uv run python exp/selective-copying/train.py --run-name d512-logu \
        --t-dist loguniform
"""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task  # noqa: F401  (binds selective task as sys.modules["task"])

assert sys.modules["task"].TASK_NAME == "selective-copying"
runpy.run_path(str(HERE.parent / "copying" / "train.py"), run_name="__main__")
