"""E5 memory-honesty figure: four-system stacked bar (correct / hallucinated / honest).

Reads the cached three-way classification from results/e5/stats.json (no API key,
no LLM) and renders the stacked bar used in the paper. Each system's column sums to
1.0: green = correct recall, red = confident-but-wrong hallucination, blue = honest
abstention. The contrast that matters: only the Linger systems (B−, B) ever abstain
instead of fabricating, and turning on governed fact-memory (B) converts most of
B−'s abstentions into correct recalls.

Run:  python -m figures.plot_memory_honesty   (or `make figures`)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: render straight to a file, never open a GUI window
import matplotlib.pyplot as plt  # noqa: E402  (must come after use("Agg"))
import numpy as np  # noqa: E402

from experiments.common.naming import CODE, PAPER, canon_keys  # noqa: E402

# parents[1]: this file is figures/plot_memory_honesty.py, so parents[0]=figures/, parents[1]=repo root.
ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "results" / "e5" / "stats.json"
OUT_PDF = ROOT / "figures" / "memory_honesty.pdf"
OUT_PNG = ROOT / "figures" / "memory_honesty.png"

ORDER = list(CODE)
SUBTITLE = {"bare": "(bare LLM)", "rag": "(bare+RAG)", "linger_no_factmem": "(abl.)", "linger_full": "(ours)"}
GREEN, RED, BLUE = "#4a9b5d", "#c0392b", "#2c6fbb"


def main() -> None:
    threeway = json.loads(STATS.read_text(encoding="utf-8"))["memory_threeway"]
    by_sys = canon_keys(threeway)  # fold any naming variant to canonical code names

    correct = [by_sys[s]["correct"]["rate"] for s in ORDER]
    halluc = [by_sys[s]["hallucinated"]["rate"] for s in ORDER]
    honest = [by_sys[s]["honest"]["rate"] for s in ORDER]

    x = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    bottom = np.zeros(len(ORDER))  # running base of the stack; each layer sits on the previous
    for vals, color, label in [
        (correct, GREEN, "Correct recall"),
        (halluc, RED, "Hallucinated"),
        (honest, BLUE, "Abstained (honest)"),
    ]:
        ax.bar(x, vals, 0.62, bottom=bottom, color=color, label=label)
        bottom = bottom + np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{PAPER[s]}\n{SUBTITLE[s]}" for s in ORDER], fontsize=6.5)
    ax.set_ylabel("Fraction of memory probes", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.5, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.20), frameon=False)
    fig.tight_layout(pad=0.4)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    n = by_sys["bare"]["correct"]["n"]
    print(f"saved {OUT_PDF.relative_to(ROOT)} and {OUT_PNG.relative_to(ROOT)} "
          f"(four-system A1/A2/B−/B, n={n}/system)")


if __name__ == "__main__":
    main()
