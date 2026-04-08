# server/app.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: F401


def main():
    """Entry point for OpenEnv multi-mode deployment."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        workers=1,
    )


if __name__ == "__main__":
    main()
