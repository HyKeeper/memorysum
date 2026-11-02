"""Command-line server entry point; the public API port is fixed at 8001."""

from __future__ import annotations

from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from memorysum.settings import RuntimeSettings


def _load_env_file() -> None:
    """Load the caller's ``.env`` without overriding exported variables.

    ``load_dotenv()`` on its own walks up from *this module's* file, which lives in
    site-packages once the project is installed — the user's own ``.env`` would be
    silently ignored. Prefer the current working directory, and fall back to the
    package-relative search only when there is no ``.env`` there.
    """
    candidate = Path.cwd() / ".env"
    if candidate.is_file():
        load_dotenv(candidate, override=False)
    else:
        load_dotenv(override=False)


def main() -> None:
    _load_env_file()
    runtime = RuntimeSettings.from_environment()
    uvicorn.run(
        "memorysum.asgi:app",
        host=runtime.host,
        port=runtime.port,
        workers=1,
        # RequestLoggingMiddleware emits access logs with UTC time and latency.
        access_log=False,
    )


if __name__ == "__main__":
    main()
