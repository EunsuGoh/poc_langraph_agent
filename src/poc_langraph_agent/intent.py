"""Intent detection chain built with LangChain structured output."""
from __future__ import annotations

import re
from typing import Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda

from .schemas import IntentPayload, RouteCandidate, SafetyMetadata

ORDER_PATTERN = re.compile(r"\bORD-[A-Za-z0-9]+\b")
_TIME_HINTS = {
    "오늘": "today",
    "어제": "yesterday",
    "지난주": "last_week",
    "지난 달": "last_month",
}


def _score_confidence(tokens: List[str]) -> float:
    if "환불" in tokens and any(token.startswith("ord-") for token in tokens):
        return 0.92
    if "환불" in tokens:
        return 0.75
    if "주문" in tokens:
        return 0.7
    return 0.25


def build_intent_chain():
    parser = PydanticOutputParser(pydantic_object=IntentPayload)

    def _predict(payload: Dict[str, object]) -> IntentPayload:
        text = str(payload["masked_input"])
        pii_types = payload.get("pii_types", [])
        tokens = [t.lower() for t in re.findall(r"[\w-]+", text)]

        intent = "qa"
        slots: Dict[str, str] = {}
        if "환불" in tokens:
            intent = "refund_request"
        elif "주문" in tokens or "배송" in tokens:
            intent = "order_status"

        order_match = ORDER_PATTERN.search(text)
        if order_match:
            slots["order_id"] = order_match.group(0)

        for hint, canonical in _TIME_HINTS.items():
            if hint in text:
                slots["time_hint"] = canonical
                break

        confidence = _score_confidence(tokens)
        reason = "환불 관련 키워드와 주문번호를 감지" if intent == "refund_request" else "주문/배송 키워드 기반 분류"

        if intent == "refund_request":
            route_candidates = [
                RouteCandidate(
                    plan_hint="refund_linear",
                    agents=["order_agent.v1", "refund_agent.v1", "response_agent.v1"],
                    est_cost="medium",
                    notes="주문 확인 후 환불 승인 흐름",
                )
            ]
        else:
            route_candidates = [
                RouteCandidate(
                    plan_hint="order_status",
                    agents=["order_agent.v1", "response_agent.v1"],
                    est_cost="low",
                    notes="주문 상태 확인 기본 흐름",
                )
            ]

        safety = SafetyMetadata(has_pii=bool(pii_types), pii_types=list(pii_types))

        data = IntentPayload(
            intent=intent,
            confidence=confidence,
            slots=slots,
            safety=safety,
            route_candidates=route_candidates,
            reason=reason,
        )
        return parser.parse(data.json())

    return RunnableLambda(_predict)
