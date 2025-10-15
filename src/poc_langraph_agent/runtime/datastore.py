"""Simple JSON-based datastore for orders and refunds."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
ORDERS_PATH = ROOT / "assets" / "orders.json"
REFUNDS_PATH = ROOT / "assets" / "refunds.json"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def get_order(order_id: str) -> Dict[str, Any] | None:
    orders = _load_json(ORDERS_PATH)
    return orders.get(order_id)


def load_refunds() -> Dict[str, Any]:
    return _load_json(REFUNDS_PATH)


def record_refund(order_id: str, action: str, notes: list[str]) -> Dict[str, Any]:
    refunds = load_refunds()
    refunds[order_id] = {
        "action": action,
        "notes": notes,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_json(REFUNDS_PATH, refunds)
    return refunds[order_id]
