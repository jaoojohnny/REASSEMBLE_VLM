"""讓 `python -m scripts.xxx` 在尚未 pip install 時也找得到 src/assembly_monitor。"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
try:
    import assembly_monitor  # noqa: F401
except ImportError:
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
