# server/app.py
# OpenEnv multi-mode deployment entry point.
# Proxies to the root app.py so the validator finds server/app.py
# while the Dockerfile continues to work with root app.py unchanged.

import sys
import os

# Add the repo root to path so we can import root app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-export the FastAPI app from root app.py
from app import app  # noqa: F401

__all__ = ["app"]
