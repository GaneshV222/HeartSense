"""Compatibility alias for the backend app package.

This project's legacy test imports use the top-level package name `app`, while the
runtime app is actually located under `backend/app`. Expose that directory as the
`app` package so both import styles work without rewriting the real backend.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.join(_ROOT, "..", "backend")
_BACKEND_APP = os.path.join(_BACKEND_ROOT, "app")

for _candidate in (_BACKEND_ROOT, _BACKEND_APP):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

__path__ = [_BACKEND_APP]
