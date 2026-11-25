"""``python -m webui`` - start the owner-only web dashboard on http://127.0.0.1:8600 (PAPER only)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from webui.server import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
