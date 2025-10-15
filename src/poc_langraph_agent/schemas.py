"""Shared Pydantic schemas for intent detection and planning."""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


class RouteCandidate(BaseModel):
    plan_hint: str = Field(..., description="Identifier for the suggested plan template")
    agents: List[str] = Field(default_factory=list, description="Ordered list of agent IDs to run")
    est_cost: Literal["low", "medium", "high"] = Field(
        "low", description="Rough qualitative estimate of cost"
    )
    notes: Optional[str] = Field(None, description="Operator-facing note about this route")


class SafetyMetadata(BaseModel):
    has_pii: bool = False
    pii_types: List[str] = Field(default_factory=list)
    prompt_injection_suspected: bool = False


class IntentSlots(BaseModel):
    order_id: Optional[str] = None
    time_hint: Optional[str] = None


class IntentPayload(BaseModel):
    intent: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    slots: IntentSlots = Field(default_factory=IntentSlots)
    safety: SafetyMetadata = Field(default_factory=SafetyMetadata)
    route_candidates: List[RouteCandidate] = Field(default_factory=list)
    reason: str

    @validator("intent")
    def intent_lowercase(cls, value: str) -> str:
        return value.strip().lower()


class OrderAgentResult(BaseModel):
    order_status: str
    refund_eligible: bool
    notes: List[str] = Field(default_factory=list)


class RefundAgentResult(BaseModel):
    refund_action: Literal["approve", "deny"]
    refund_id: str
    notes: List[str] = Field(default_factory=list)


class ResponseAgentResult(BaseModel):
    message: str


class PlannerNode(BaseModel):
    id: str
    agent: str
    input_key: str = "payload"
    output_key: str = "payload"
    timeout_ms: int = Field(1000, ge=50, le=60000)
    max_retries: int = Field(2, ge=0, le=5)


class Plan(BaseModel):
    plan_id: str
    description: str
    nodes: List[PlannerNode]
    terminal_key: str = "payload"

    @validator("nodes")
    def ensure_linear(cls, value: List[PlannerNode]) -> List[PlannerNode]:
        if not value:
            raise ValueError("Plan must contain at least one node")
        seen = set()
        for node in value:
            if node.id in seen:
                raise ValueError("Duplicate node id detected")
            seen.add(node.id)
        return value


class RouterState(BaseModel):
    raw_input: str
    masked_input: str
    intent: Optional[IntentPayload] = None
    plan: Optional[Plan] = None
    payload: Dict[str, object] = Field(default_factory=dict)
    transcript: List[str] = Field(default_factory=list)
    error: Optional[str] = None
