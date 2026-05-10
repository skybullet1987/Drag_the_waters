"""Repo-root conftest — adds workspace root to sys.path.

Lets `from backtest_audit.X import Y` and `from Pulse.X import Y` work
in both pytest and direct script execution without needing PYTHONPATH.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
