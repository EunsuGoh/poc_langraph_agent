"""Sequential executor with retries and backoff."""
from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Dict

from ..schemas import Plan
from ..tools.nodes import AgentExecutionError, get_node
from .safety import jitter_backoff


class Executor:
    def __init__(self, base_delay: float = 0.25):
        self.base_delay = base_delay

    def _run_call(self, node_id: str, payload: Dict[str, Any], timeout_ms: int) -> Dict[str, Any]:
        node_spec = get_node(node_id)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(node_spec.handler, payload)
            try:
                return future.result(timeout=timeout_ms / 1000)
            except concurrent.futures.TimeoutError as exc:
                future.cancel()
                raise AgentExecutionError(f"Node {node_id} timed out") from exc

    def execute(self, plan: Plan, initial_payload: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(initial_payload)
        transcript = payload.setdefault("trace", [])
        for node in plan.nodes:
            attempts = 0
            last_error: Exception | None = None
            while attempts <= node.max_retries:
                try:
                    transcript.append({"node": node.agent, "attempt": attempts + 1})
                    payload = self._run_call(node.agent, payload, node.timeout_ms)
                    break
                except AgentExecutionError as exc:
                    last_error = exc
                    attempts += 1
                    if attempts > node.max_retries:
                        raise
                    delay = jitter_backoff(self.base_delay, attempts)
                    transcript.append(
                        {
                            "node": node.agent,
                            "error": str(exc),
                            "retry_in": round(delay, 3),
                            "attempt": attempts,
                        }
                    )
                    time.sleep(delay)
        return payload
