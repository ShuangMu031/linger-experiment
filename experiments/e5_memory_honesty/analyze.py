"""E5 memory-honesty analysis. Two modes:

  (default, --from-cache)  OFFLINE. Read results/e5/stats.json and recompute every *statistic*
       from the cached three-way counts — per-class rate + Wilson 95% CI, the between-system
       two-proportion z-tests with Holm-Bonferroni — then echo the cached seed-level mean+/-SD
       and paired-bootstrap diffs. No API key. This is what `make analyze` runs.

  (--recompute)  END-TO-END. Read the raw dialogue rows (results/e5/{bare,rag,
       linger_no_factmem,linger_full}.json produced by `make e5`), call the cross-family judge
       on every recall (correct / hallucinated / honest), aggregate into results/e5/stats.json,
       then render. Needs an API key.

All system keys (in data and in stats.json) are canonical CODE names; PAPER names (A1/A2/B−/B)
appear only when printing. See experiments.common.naming.

Run:  python -m experiments.e5_memory_honesty.analyze              (offline)
      python -m experiments.e5_memory_honesty.analyze --recompute  (after make e5; needs API)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from experiments.common.naming import CODE, PAPER, canon_keys, ordered_codes
from experiments.common.stats import (
    holm_bonferroni,
    mean_sd,
    paired_bootstrap_diff,
    two_proportion_z,
    wilson_ci,
)

# parents[2]: experiments/e5_memory_honesty/analyze.py -> [2] = repo root.
ROOT = Path(__file__).resolve().parents[2]
E5_DIR = ROOT / "results" / "e5"
STATS = E5_DIR / "stats.json"
CATS = ("correct", "hallucinated", "honest")


# ---------------------------------------------------------------------------
# --recompute: raw dialogue rows -> judged stats.json (END-TO-END, needs API)
# ---------------------------------------------------------------------------
def recompute_stats() -> dict:
    from experiments.common.systems import build_llm_config
    from experiments.common.judges import classify_memory
    from experiments.e5_memory_honesty.probes import MEMORY_FACTS

    cfg = build_llm_config()
    runs = {}
    for code in CODE:
        f = E5_DIR / f"{code}.json"
        if f.exists():
            runs.update(json.loads(f.read_text(encoding="utf-8")).get("runs", {}))
    if not runs:
        raise SystemExit(f"no raw runs in {E5_DIR} — run `make e5` first")

    # Aggregate recall rows by system; exclude empty-reply turns (engineering failures, not abstentions).
    by_sys = defaultdict(list)
    for key, r in runs.items():
        sysname = key.split("_seed")[0]
        for row in r["rows"]:
            if row["kind"] == "recall" and (row.get("reply") or "").strip():
                by_sys[sysname].append((key, row))

    class_counts, seed_counts = {}, defaultdict(lambda: defaultdict(Counter))
    for sysname, items in by_sys.items():
        c = Counter()
        for seedkey, row in items:
            fact = MEMORY_FACTS[row["fact_idx"]]
            cl = classify_memory(cfg, row["user"], fact["plant"], fact["key"], row["reply"] or "")
            c[cl] += 1
            seed_counts[sysname][seedkey][cl] += 1
        class_counts[sysname] = {"counts": dict(c), "n": sum(c.values())}
        print(f"[{PAPER.get(sysname, sysname)}] n={class_counts[sysname]['n']} {dict(c)}", flush=True)

    present = ordered_codes(class_counts)
    threeway = {s: {cat: {"rate": round(class_counts[s]["counts"].get(cat, 0) / (class_counts[s]["n"] or 1), 3),
                          "ci95": wilson_ci(class_counts[s]["counts"].get(cat, 0), class_counts[s]["n"]),
                          "k": class_counts[s]["counts"].get(cat, 0), "n": class_counts[s]["n"]}
                    for cat in CATS} for s in present}

    tests = {}
    for cat in ("hallucinated", "honest"):
        pairs = []
        for a, b in combinations(present, 2):
            ka, na = class_counts[a]["counts"].get(cat, 0), class_counts[a]["n"]
            kb, nb = class_counts[b]["counts"].get(cat, 0), class_counts[b]["n"]
            z, p = two_proportion_z(ka, na, kb, nb)
            pairs.append((f"{PAPER[a]} vs {PAPER[b]} (z={z})", p))
        tests[cat] = holm_bonferroni(pairs)

    # aux: identity / boundary / judge-mean from the rows
    aux = {}
    raw_aux = defaultdict(lambda: {"id_kept": 0, "id_n": 0, "bd_viol": 0, "bd_n": 0, "judge": []})
    for key, r in runs.items():
        sysname = key.split("_seed")[0]
        for row in r["rows"]:
            if row["kind"] == "identity":
                raw_aux[sysname]["id_n"] += 1
                raw_aux[sysname]["id_kept"] += int(row.get("kept", False))
            elif row["kind"] == "boundary":
                raw_aux[sysname]["bd_n"] += 1
                raw_aux[sysname]["bd_viol"] += int(row.get("violated", False))
        raw_aux[sysname]["judge"] += [s for _, s in r.get("judge_curve", [])]
    for s in present:
        a = raw_aux[s]
        aux[s] = {
            "identity_retention": round(a["id_kept"] / a["id_n"], 3) if a["id_n"] else None,
            "boundary_violation_rate": round(a["bd_viol"] / a["bd_n"], 3) if a["bd_n"] else None,
            "judge_mean": round(sum(a["judge"]) / len(a["judge"]), 1) if a["judge"] else None,
        }

    # per-seed rates -> seed_summary (mean+/-sd) + paired bootstrap on the seed-level rates
    seed_rates = {}
    for s in present:
        acc = {cat: [] for cat in CATS}
        for _seedkey, cc in sorted(seed_counts[s].items()):
            n = sum(cc.values()) or 1
            for cat in CATS:
                acc[cat].append(cc.get(cat, 0) / n)
        seed_rates[s] = acc
    seed_summary = {s: {cat: {"mean": mean_sd(v[cat])[0], "sd": mean_sd(v[cat])[1]} for cat in CATS}
                    for s, v in seed_rates.items()}
    boot = {}
    if "linger_full" in seed_rates and "linger_no_factmem" in seed_rates:
        boot["correct: B vs B−"] = paired_bootstrap_diff(
            seed_rates["linger_full"]["correct"], seed_rates["linger_no_factmem"]["correct"], seed=0)
    if "linger_full" in seed_rates and "rag" in seed_rates:
        boot["correct: B vs A2"] = paired_bootstrap_diff(
            seed_rates["linger_full"]["correct"], seed_rates["rag"]["correct"], seed=0)
    if "linger_no_factmem" in seed_rates and "bare" in seed_rates:
        boot["honest: B− vs A1"] = paired_bootstrap_diff(
            seed_rates["linger_no_factmem"]["honest"], seed_rates["bare"]["honest"], seed=0)

    stats = {"memory_threeway": threeway, "between_system_tests": tests, "aux": aux,
             "class_counts": {s: class_counts[s] for s in present},
             "seed_summary": seed_summary, "bootstrap_diff": boot}
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {STATS.relative_to(ROOT)}", flush=True)
    return stats


# ---------------------------------------------------------------------------
# rendering (shared by both modes)
# ---------------------------------------------------------------------------
def render(stats: dict) -> None:
    counts = canon_keys(stats["class_counts"])
    order = ordered_codes(counts)

    print("=== E5 memory three-way: rate [95% Wilson CI] ===")
    table = {}
    for s in order:
        n = counts[s]["n"]
        c = counts[s]["counts"]
        table[s] = {cat: (c.get(cat, 0), n) for cat in CATS}
        parts = [f"{cat:<12} {c.get(cat, 0) / n:.2f} [{wilson_ci(c.get(cat, 0), n)[0]:.2f},"
                 f"{wilson_ci(c.get(cat, 0), n)[1]:.2f}]" for cat in CATS]
        print(f"  {PAPER[s]:<4} (n={n})  " + " | ".join(parts))

    for cat in ("hallucinated", "honest"):
        pairs = []
        for a, b in combinations(order, 2):
            (ka, na), (kb, nb) = table[a][cat], table[b][cat]
            z, p = two_proportion_z(ka, na, kb, nb)
            pairs.append((f"{PAPER[a]} vs {PAPER[b]} (z={z})", p))
        print(f"\n=== between-system {cat} rate: two-proportion z, Holm-Bonferroni ===")
        for label, p, p_adj, sig in holm_bonferroni(pairs):
            print(f"  {label:<22} p={p:.4g}  p_adj={p_adj:.4g}  {'*' if sig else ''}")

    seed_summary = canon_keys(stats.get("seed_summary", {}))
    if seed_summary:
        print("\n=== seed-level mean+/-SD ===")
        for s in ordered_codes(seed_summary):
            d = seed_summary[s]
            cells = " | ".join(f"{cat} {d[cat]['mean']:.2f}+/-{d[cat]['sd']:.3f}" for cat in CATS if cat in d)
            print(f"  {PAPER[s]:<4} {cells}")

    if stats.get("bootstrap_diff"):
        print("\n=== paired bootstrap (rate diff, 95% CI; excludes 0 -> significant) ===")
        for label, (lo, hi) in stats["bootstrap_diff"].items():
            print(f"  {label:<16} [{lo}, {hi}]  {'*' if (lo > 0 or hi < 0) else ''}")

    aux = canon_keys(stats.get("aux", {}))
    if aux:
        print("\n=== aux (persona side-metrics during the memory run) ===")
        for s in ordered_codes(aux):
            a = aux[s]
            print(f"  {PAPER[s]:<4} identity_retention={a['identity_retention']} "
                  f"boundary_violation={a['boundary_violation_rate']} judge_mean={a['judge_mean']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="E5 memory-honesty analysis")
    ap.add_argument("--from-cache", action="store_true",
                    help="read results/e5/stats.json (default; offline, no API)")
    ap.add_argument("--recompute", action="store_true",
                    help="re-judge raw runs into stats.json, then render (needs API key)")
    args = ap.parse_args()

    if args.recompute:
        stats = recompute_stats()
    else:
        if not STATS.exists():
            raise SystemExit(f"missing {STATS}\n  -> run `make e5` then `... analyze --recompute`, "
                             "or restore the cached file")
        stats = json.loads(STATS.read_text(encoding="utf-8"))
    render(stats)


if __name__ == "__main__":
    main()
