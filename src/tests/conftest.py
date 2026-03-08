"""
pytest configuration for the Anti-Fraud Framework test suite.

Adds src/ to sys.path so all module imports work when running
`pytest src/tests/` from the project root.
"""

import sys
from pathlib import Path

# Ensure src/ is importable
_SRC = Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
