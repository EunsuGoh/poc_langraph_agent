"""LangGraph router wiring pre/post safety, intent, planning, and execution."""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from ..intent import build_intent_chain
from ..schemas import IntentPayload, Plan, RouterState
from ..tools.nodes import append_agent_trace
from .executor import Executor
from .planner import build_planner_chain
from .safety import contains_forbidden_term, mask_pii


def _format_plan_tree(plan: Plan) -> str:
    lines = [f"plan_id={plan.plan_id}"]
    total = len(plan.nodes)
    for index, node in enumerate(plan.nodes):
        prefix = "|-" if index < total - 1 else "\\-"
        lines.append(f"{prefix} {node.id}: {node.agent}")
    return "\n".join(lines)


def _preprocess_node(state: Dict[str, Any]) -> Dict[str, Any]:
    masked, pii = mask_pii(state["raw_input"])
    state.update({"masked_input": masked, "pii_types": list(pii)})
    state.setdefault("transcript", []).append("pre:mask_pii")
    return state


def _intent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    intent_chain = build_intent_chain()
    intent: IntentPayload = intent_chain.invoke(
        {"masked_input": state["masked_input"], "pii_types": state.get("pii_types", [])}
    )
    state["intent"] = intent
    state.setdefault("transcript", []).append("intent:structured_output")
    return state


def _plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    planner_chain = build_planner_chain()
    plan: Plan = planner_chain.invoke({"intent": state["intent"]})
    state["plan"] = plan
    state.setdefault("transcript", []).append(plan.plan_id)
    payload = state.setdefault("payload", {})
    append_agent_trace(
        payload,
        agent_id="plan_agent.v1",
        label="PLAN",
        message="그래프 계획 생성 완료",
        dag=_format_plan_tree(plan),
        plan=plan.dict(),
    )
    return state


def _executor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    executor = Executor()
    payload = state.get("payload", {})
    payload.setdefault("query", state["masked_input"])

    intent: IntentPayload | None = state.get("intent")
    if intent is not None:
        payload.setdefault("intent", intent.dict())
        payload.setdefault("slots", intent.slots.dict())
        if intent.slots.order_id:
            payload.setdefault("order_id", intent.slots.order_id)

    result = executor.execute(state["plan"], payload)
    state["payload"] = result
    state.setdefault("transcript", []).append("executor:done")
    return state


def _postprocess_node(state: Dict[str, Any]) -> Dict[str, Any]:
    response = state.get("payload", {}).get("response", "")
    if contains_forbidden_term(response):
        state["error"] = "정책 위반"
        state.setdefault("transcript", []).append("post:policy_violation")
        return state
    state.setdefault("transcript", []).append("post:ok")
    return state


def build_router_graph():
    graph = StateGraph(dict)
    graph.add_node("pre", _preprocess_node)
    graph.add_node("intent", _intent_node)
    graph.add_node("plan", _plan_node)
    graph.add_node("executor", _executor_node)
    graph.add_node("post", _postprocess_node)

    graph.set_entry_point("pre")
    graph.add_edge("pre", "intent")
    graph.add_edge("intent", "plan")
    graph.add_edge("plan", "executor")
    graph.add_edge("executor", "post")
    graph.add_edge("post", END)

    return graph.compile()


def run_router(user_input: str) -> RouterState:
    graph = build_router_graph()
    state: Dict[str, Any] = {"raw_input": user_input, "masked_input": user_input}
    result = graph.invoke(state)
    return RouterState.parse_obj(result)
