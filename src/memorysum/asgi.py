"""Uvicorn import target; kept separate so importing API schemas has no side effects."""

from memorysum.api.app import create_app

app = create_app()
