"""Safety utilities for the MCP router."""
import random
import re
from typing import List, Tuple

_PII_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("ssn", re.compile(r"(?P<mask>\b\d{6}-?\d{7}\b)")),
    ("phone", re.compile(r"(?P<mask>\b01[016789]-?\d{3,4}-?\d{4}\b)")),
    ("email", re.compile(r"(?P<mask>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")),
)

_FORBIDDEN_TERMS = {"금지어", "외부유출", "파괴"}


def mask_pii(text: str) -> Tuple[str, List[str]]:
    """Mask known PII patterns with placeholder tokens."""
    detected = []
    masked = text
    for label, pattern in _PII_PATTERNS:
        if not pattern.search(masked):
            continue
        detected.append(label)
        masked = pattern.sub("<PII>", masked)
    return masked, detected


def contains_forbidden_term(text: str) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in _FORBIDDEN_TERMS)


def jitter_backoff(base_delay: float, attempt: int) -> float:
    """Return exponential backoff delay with jitter."""
    jitter = random.uniform(0, base_delay / 2)
    return base_delay * (2 ** attempt) + jitter
