"""Single source of truth for the four-system naming. Import this everywhere.

Rule: the CODE name is canonical — it is what every data key, JSON key and variable
uses. The PAPER name is for display only (tables, figures, printed labels). ALIAS
folds every legacy variant (A1/A2/B0/B+/B-/B−/B) back to the canonical code name when
reading older caches, so no other module needs to know about historical spellings.

  code name          paper name   what it is
  bare               A1           bare LLM + persona + sliding window
  rag                A2           A1 + BM25 retrieval
  linger_no_factmem  B−           full Linger, user-fact memory OFF (ablation)
  linger_full        B            full Linger, user-fact memory ON (ours)
"""
from __future__ import annotations

from typing import List

# Canonical keys, ordered A1..B. Use for data keys, JSON keys, variables.
CODE = ("bare", "rag", "linger_no_factmem", "linger_full")

# Display names (tables / figures only). Unicode minus "B−" to match the paper.
PAPER = {
    "bare": "A1",
    "rag": "A2",
    "linger_no_factmem": "B−",
    "linger_full": "B",
}

# Every legacy / display variant -> canonical code name (for reading old caches).
# Note "B" = linger_full (ours, full), "B−"/"B-"/"B0" = linger_no_factmem (ablation).
ALIAS = {
    "bare": "bare", "A1": "bare",
    "rag": "rag", "A2": "rag",
    "linger_no_factmem": "linger_no_factmem", "B0": "linger_no_factmem",
    "B-": "linger_no_factmem", "B−": "linger_no_factmem",
    "linger_full": "linger_full", "B+": "linger_full", "B": "linger_full",
}


def to_code(name: str) -> str:
    """Fold any naming variant to its canonical code name."""
    return ALIAS[name]


def to_paper(name: str) -> str:
    """Map any naming variant to its paper display name (A1 / A2 / B− / B)."""
    return PAPER[ALIAS[name]]


def canon_keys(block: dict) -> dict:
    """Re-key a {system: value} dict to canonical code names, regardless of input spelling."""
    return {to_code(k): v for k, v in block.items()}


def ordered_codes(present) -> List[str]:
    """The subset of CODE that appears in `present`, in canonical A1..B order."""
    present = set(present)
    return [c for c in CODE if c in present]


if __name__ == "__main__":
    assert to_code("B+") == "linger_full" and to_code("B0") == "linger_no_factmem"
    assert to_code("B−") == "linger_no_factmem" and to_code("B-") == "linger_no_factmem"
    assert to_paper("B0") == "B−" and to_paper("linger_full") == "B"
    assert canon_keys({"A1": 1, "B+": 2}) == {"bare": 1, "linger_full": 2}
    assert ordered_codes({"linger_full", "bare"}) == ["bare", "linger_full"]
    print("NAMING SELFTEST PASSED")
