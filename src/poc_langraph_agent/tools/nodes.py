"""Registry of Gemini-backed agent nodes for the MCP executor."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage

from ..runtime.datastore import get_order, load_refunds, record_refund
from ..runtime.llm import MissingAPIKeyError, get_gemini
from ..runtime.prompts import load_prompt
from ..schemas import OrderAgentResult, RefundAgentResult, ResponseAgentResult


class AgentExecutionError(RuntimeError):
    """Raised when an agent node fails irrecoverably."""


@dataclass
class NodeSpec:
    """Runtime metadata for an agent node."""

    name: str
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]


def _format_json(data: Any) -> str:
    if not data:
        return "{}"
    return json.dumps(data, ensure_ascii=False, indent=2)


def _extract_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, BaseMessage):
        value = message.content
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for block in value:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            if parts:
                return "".join(parts)
    if hasattr(message, "text"):
        return getattr(message, "text")
    return str(message)


def _call_structured_agent(
    system_name: str,
    user_name: str,
    output_schema: Type,
    variables: Dict[str, Any],
):
    system_prompt = load_prompt(system_name)
    user_prompt = load_prompt(user_name)
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", user_prompt)]
    )
    try:
        llm = get_gemini()
    except MissingAPIKeyError as exc:
        raise AgentExecutionError(str(exc)) from exc
    chain = prompt | llm
    try:
        response = chain.invoke(variables)
    except Exception:
        # One-off fallback to a more widely available model if initial request fails
        llm = get_gemini("gemini-2.5-flash")
        chain = prompt | llm
        response = chain.invoke(variables)
    text = _extract_text(response).strip()
    if text.startswith("```"):
        text = text.strip("`\n\t ")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentExecutionError(f"Failed to decode JSON: {text}") from exc
    try:
        return output_schema.model_validate(payload)
    except Exception as exc:
        raise AgentExecutionError(f"Validation error: {payload}") from exc


def order_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    order_id = payload.get("order_id") or payload.get("slots", {}).get("order_id")
    query = payload.get("query", "")
    refunds = load_refunds()
    order_record = get_order(order_id) if order_id else None
    result: OrderAgentResult = _call_structured_agent(
        "order_agent_system",
        "order_agent_user",
        OrderAgentResult,
        {
            "user_query": query,
            "order_id": order_id or "UNKNOWN",
            "order_record": _format_json(order_record),
            "refund_record": _format_json(refunds.get(order_id, {})),
        },
    )
    payload["order_id"] = order_id
    payload["order"] = {
        "order_id": order_id,
        "record": order_record,
        "analysis": result.dict(),
    }
    payload["order_agent_result"] = result.dict()
    payload.setdefault("trace", []).append({"node": "order_agent.v1", "status": result.order_status})
    return payload


def refund_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    order_id = payload.get("order_id")
    if not order_id:
        raise AgentExecutionError("order_id missing in payload")
    order_result = payload.get("order_agent_result")
    if not order_result:
        raise AgentExecutionError("order agent result missing")

    refunds = load_refunds()
    result: RefundAgentResult = _call_structured_agent(
        "refund_agent_system",
        "refund_agent_user",
        RefundAgentResult,
        {
            "user_query": payload.get("query", ""),
            "order_agent_result": _format_json(order_result),
            "refund_record": _format_json(refunds.get(order_id, {})),
        },
    )

    refund_id = result.refund_id
    if not refund_id or refund_id.lower() in {"none", "n/a"}:
        refund_id = f"REF-{uuid.uuid4().hex[:8].upper()}"
        result = RefundAgentResult(
            refund_action=result.refund_action,
            refund_id=refund_id,
            notes=result.notes,
        )

    record_refund(order_id, result.refund_action, result.notes)

    payload["refund"] = {
        "order_id": order_id,
        "result": result.dict(),
    }
    payload["refund_agent_result"] = result.dict()
    payload.setdefault("trace", []).append(
        {"node": "refund_agent.v1", "action": result.refund_action, "refund_id": refund_id}
    )
    return payload


def response_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    order_result = payload.get("order_agent_result") or {}
    refund_result = payload.get("refund_agent_result") or {}
    result: ResponseAgentResult = _call_structured_agent(
        "response_agent_system",
        "response_agent_user",
        ResponseAgentResult,
        {
            "user_query": payload.get("query", ""),
            "order_agent_result": _format_json(order_result),
            "refund_agent_result": _format_json(refund_result),
        },
    )
    payload["response"] = result.message
    payload.setdefault("trace", []).append({"node": "response_agent.v1"})
    return payload


_NODE_FACTORY = {
    "order_agent.v1": NodeSpec("order_agent.v1", order_agent),
    "refund_agent.v1": NodeSpec("refund_agent.v1", refund_agent),
    "response_agent.v1": NodeSpec("response_agent.v1", response_agent),
}


def get_node(agent_id: str) -> NodeSpec:
    try:
        return _NODE_FACTORY[agent_id]
    except KeyError as exc:
        raise AgentExecutionError(f"Unknown agent: {agent_id}") from exc
