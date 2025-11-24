"""
Return binary rewards by extracting final answer from \boxed{}
"""

import re
from typing import List, Optional, Any

def _extract_boxed_all(text: str) -> List[str]:
    """
    Extract all occurrences of \boxed{...} from TEXT (single backslash),
    using brace matching to support nested braces.
    """
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
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1

        if depth == 0:
            results.append(text[content_start : j - 1])
            i = j
        else:
            break  # unmatched braces -> stop

    return results


def _normalize_answer(ans: Optional[str]) -> str:
    """
    Light normalization (verl-style tolerant):
    - strip spaces
    - remove outer $...$
    - collapse whitespace
    """
    if ans is None:
        return ""
    s = ans.strip()
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    s = re.sub(r"\s+", " ", s)
    return s


def my_reward_fn(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
) -> float:
    """
    verl-compatible reward function.

    Rule:
    1) Only extract answer from \boxed{...} (single backslash in output text).
    2) Take the LAST boxed as prediction.
    3) If extracted and matches ground_truth after light normalize => reward 1.0
       else => reward 0.0
    """
    # verl often only looks at tail to avoid prompt noise
    tail = solution_str[-300:] if solution_str is not None else ""

    boxed = _extract_boxed_all(tail)
    if not boxed:
        return 0.0

    pred = _normalize_answer(boxed[-1])
    gt = _normalize_answer(ground_truth)

    return 1.0 if pred == gt else 0.0
