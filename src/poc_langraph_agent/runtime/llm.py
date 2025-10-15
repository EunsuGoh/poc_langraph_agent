"""Shared helpers for interacting with Gemini models."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_MODEL = "gemini-2.5-flash"
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Ensure .env is loaded once at import time for any entrypoint
dotenv_path = find_dotenv(usecwd=True) or PROJECT_ROOT / ".env"
if dotenv_path:
    # Prioritize .env values to avoid empty env vars shadowing them
    load_dotenv(dotenv_path, override=True)


class MissingAPIKeyError(RuntimeError):
    """Raised when Google Generative AI API key is not configured."""


@lru_cache(maxsize=1)
def get_gemini(model: str | None = None) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("GOOGLE_API_KEY not set. Update your .env file.")
    # Normalize model name to the canonical format expected by the Google GenAI SDK
    configured = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
    # Strip unsupported alias suffixes (e.g., "-latest")
    if configured.endswith("-latest"):
        configured = configured[: -len("-latest")]
    # Use Makersuite-style names (no "models/" prefix) to avoid v1beta 404s
    model_name = configured
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.1
    )
