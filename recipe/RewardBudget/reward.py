"""
Binary reward for math-style answers wrapped in \\boxed{}.
"""

import re
from typing import Any, List, Optional


def _extract_boxed_all(text: str) -> List[str]:
    """Extract all \\boxed{...} spans, including spans with nested braces."""
    results = []
    key = r"\boxed{"
    i, n = 0, len(text)

    while i < n:
        start = text.find(key, i)
        if start == -1:
            break

        j = start + len(key)
        depth = 1
        content_start = j

        while j < n and depth > 0:
            char = text[j]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            j += 1

        if depth == 0:
            results.append(text[content_start : j - 1])
            i = j
        else:
            break

    return results


def _normalize_answer(answer: Optional[str]) -> str:
    if answer is None:
        return ""
    normalized = answer.strip()
    if len(normalized) >= 2 and normalized[0] == "$" and normalized[-1] == "$":
        normalized = normalized[1:-1].strip()
    return re.sub(r"\s+", " ", normalized)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
    **kwargs,
) -> float:
    """Return 1.0 when the last boxed answer matches the ground truth."""
    tail = solution_str[-300:] if solution_str is not None else ""
    boxed = _extract_boxed_all(tail)
    if not boxed:
        return 0.0

    pred = _normalize_answer(boxed[-1])
    gt = _normalize_answer(ground_truth)
    return 1.0 if pred == gt else 0.0
