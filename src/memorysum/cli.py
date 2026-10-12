"""Command-line server entry point; the public API port is fixed at 8001."""

from __future__ import annotations

import uvicorn
from dotenv import load_dotenv

from memorysum.settings import RuntimeSettings


def main() -> None:
    load_dotenv(override=False)
    runtime = RuntimeSettings.from_environment()
    uvicorn.run(
        "memorysum.asgi:app",
        host=runtime.host,
        port=runtime.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
