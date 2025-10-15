"""Utility script to run the router with sample input."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from poc_langraph_agent.runtime.router import run_router


if __name__ == "__main__":
    dotenv_path = find_dotenv(usecwd=True) or ROOT / ".env"
    load_dotenv(dotenv_path)
    sample = "환불 요청합니다. 주문번호는 ORD-39422 이고 지난주 결제했습니다."
    result = run_router(sample)
    print(json.dumps(result.dict(), ensure_ascii=False, indent=2))
