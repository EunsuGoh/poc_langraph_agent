"""Simple JSON-based datastore for orders and refunds."""
from __future__ import annotations

import json
import uuid
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


def load_orders() -> Dict[str, Any]:
    data = _load_json(ORDERS_PATH)
    if isinstance(data, list):
        orders: Dict[str, Any] = {}
        for entry in data:
            order_id = entry.get("order_id")
            if not order_id:
                continue
            record = {key: value for key, value in entry.items() if key != "order_id"}
            orders[order_id] = record
        return orders
    if isinstance(data, dict):
        return data
    return {}


def get_order(order_id: str) -> Dict[str, Any] | None:
    orders = load_orders()
    return orders.get(order_id)


def load_refunds() -> Dict[str, Any]:
    data = _load_json(REFUNDS_PATH)
    if isinstance(data, list):
        refunds: Dict[str, Any] = {}
        for entry in data:
            order_id = entry.get("order_id")
            if not order_id:
                continue
            refunds[order_id] = {key: value for key, value in entry.items() if key != "order_id"}
        return refunds
    if isinstance(data, dict):
        return data
    return {}


def record_refund(order_id: str, action: str, notes: list[str]) -> Dict[str, Any]:
    refunds = load_refunds()
    refunds[order_id] = {
        "action": action,
        "notes": notes,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    refund_list = [{"order_id": oid, **record} for oid, record in refunds.items()]
    _write_json(REFUNDS_PATH, refund_list)
    return refunds[order_id]


def generate_order_id(existing: Dict[str, Any] | None = None) -> str:
    orders = existing or load_orders()
    if isinstance(orders, list):
        orders = {
            entry.get("order_id"): entry
            for entry in orders
            if isinstance(entry, dict) and entry.get("order_id")
        }
    while True:
        candidate = f"ORD-{uuid.uuid4().hex[:6].upper()}"
        if candidate not in orders:
            return candidate


def record_order(order_id: str, order_data: Dict[str, Any]) -> Dict[str, Any]:
    orders = load_orders()
    orders[order_id] = order_data
    order_list = [{"order_id": oid, **record} for oid, record in orders.items()]
    _write_json(ORDERS_PATH, order_list)
    return orders[order_id]
