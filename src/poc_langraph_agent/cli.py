"""Simple CLI entry point for the MCP router."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .runtime.router import run_router

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    dotenv_path = find_dotenv(usecwd=True) or PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path, override=True)

    parser = argparse.ArgumentParser(description="Run the MCP router against a user input")
    parser.add_argument("prompt", nargs="?", help="User utterance to route")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    user_input = args.prompt or input("사용자 요청: ")

    result = run_router(user_input)
    if args.pretty:
        print(json.dumps(result.dict(), ensure_ascii=False, indent=2))
    else:
        print(result.json(ensure_ascii=False))


if __name__ == "__main__":
    main()
