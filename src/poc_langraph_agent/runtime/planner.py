"""Planner LLM stub that produces linear DAG plans."""
from __future__ import annotations

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda

from ..schemas import IntentPayload, Plan, PlannerNode


def _plan_from_intent(intent: IntentPayload) -> Plan:
    candidate = intent.route_candidates[0]
    nodes = []
    for index, agent_id in enumerate(candidate.agents):
        nodes.append(
            PlannerNode(
                id=f"step_{index+1}",
                agent=agent_id,
                input_key="payload",
                output_key="payload",
                timeout_ms=8000 if "response_agent" not in agent_id else 12000,
                max_retries=2,
            )
        )
    return Plan(
        plan_id=candidate.plan_hint,
        description=f"Execute {candidate.plan_hint} for intent {intent.intent}",
        nodes=nodes,
    )


def build_planner_chain():
    parser = PydanticOutputParser(pydantic_object=Plan)

    def _predict(data):
        intent = data["intent"]
        if not isinstance(intent, IntentPayload):
            intent = IntentPayload.parse_raw(intent)
        plan = _plan_from_intent(intent)
        return parser.parse(plan.json())

    return RunnableLambda(_predict)
