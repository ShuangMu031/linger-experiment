"""E6 persona-consistency analysis. Two modes:

  (default)      OFFLINE. Render results/e6/stats.json and recompute the seed-level paired
                 bootstrap (A1 - B) from the cached per-seed rates. No API key. `make analyze`.

  (--recompute)  END-TO-END. Read the raw rows (results/e6/persona.json from `make e6`), call
                 the cross-family judge per script (held/violated/ambiguous + in_character/ooc +
                 break-turn), aggregate into results/e6/stats.json, then render. Needs an API key.

All system keys (in data and in stats.json) are canonical CODE names; PAPER names (A1/A2/B−/B)
appear only when printing. See experiments.common.naming.

Run:  python -m experiments.e6_persona_consistency.analyze              (offline)
      python -m experiments.e6_persona_consistency.analyze --recompute  (after make e6; needs API)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiments.common.naming import CODE, PAPER, canon_keys, to_code
from experiments.common.stats import mean_sd, paired_bootstrap_diff

# parents[2]: experiments/e6_persona_consistency/analyze.py -> [2] = repo root.
ROOT = Path(__file__).resolve().parents[2]
E6_DIR = ROOT / "results" / "e6"
STATS = E6_DIR / "stats.json"
PERSONA = E6_DIR / "persona.json"

TIERS = ("mild", "hard")
DIMS = ("identity_override", "relationship_escalation", "ai_disclosure",
        "role_play", "ooc_induction", "mixed")


def _transcript(rows, upto=None):
    return "\n".join(f"用户: {r['user']}\n鹿溪: {r['reply']}" for r in (rows[:upto] if upto else rows))


# ---------------------------------------------------------------------------
# --recompute: raw rows -> judged stats.json (END-TO-END, needs API)
# ---------------------------------------------------------------------------
def recompute_stats() -> dict:
    from experiments.common.systems import build_llm_config
    from experiments.common.judges import judge_ooc, judge_violation

    if not PERSONA.exists():
        raise SystemExit(f"no raw rows at {PERSONA} — run `make e6` first")
    cfg = build_llm_config()
    runs = json.loads(PERSONA.read_text(encoding="utf-8")).get("runs", {})

    agg = defaultdict(lambda: defaultdict(list))   # (tier, code) -> {"viol":[per-seed], "ooc":[...], "break":[...]}
    dimcnt = defaultdict(Counter)                  # (tier, code, dim) -> Counter(class)
    for key, r in runs.items():
        code = to_code(key.split("_seed")[0])
        seed_bucket = defaultdict(lambda: {"viol": 0, "ooc": 0, "n": 0, "break": []})
        for sc in r["scripts"]:
            rows = [x for x in sc["rows"] if (x.get("reply") or "").strip()]
            if not rows:
                continue
            tier = sc["tier"]
            full_t = _transcript(rows)
            vcls = judge_violation(cfg, sc["goal"], full_t)
            occls = judge_ooc(cfg, full_t)
            bt = 6
            for ti in range(1, len(rows) + 1):
                if judge_violation(cfg, sc["goal"], _transcript(rows, upto=ti)) == "violated":
                    bt = ti
                    break
            b = seed_bucket[tier]
            b["n"] += 1
            b["viol"] += int(vcls == "violated")
            b["ooc"] += int(occls == "ooc")
            b["break"].append(bt)
            dimcnt[(tier, code, sc["dim"])][vcls] += 1
        for tier, b in seed_bucket.items():
            if b["n"]:
                agg[(tier, code)]["viol"].append(b["viol"] / b["n"])
                agg[(tier, code)]["ooc"].append(b["ooc"] / b["n"])
                agg[(tier, code)]["break"].append(sum(b["break"]) / len(b["break"]))

    report = {}
    for (tier, code), d in sorted(agg.items()):
        vm, vsd = mean_sd(d["viol"])
        om, osd = mean_sd(d["ooc"])
        bm, _ = mean_sd(d["break"])
        report.setdefault(tier, {})[code] = {
            "overall_violation": {"mean": vm, "sd": vsd, "per_seed": [round(x, 3) for x in d["viol"]]},
            "ooc_rate": {"mean": om, "sd": osd, "per_seed": [round(x, 3) for x in d["ooc"]]},
            "mean_break_turn": bm,
        }
    boot = {}
    for tier in TIERS:
        a1 = agg.get((tier, "bare"), {}).get("viol", [])
        bb = agg.get((tier, "linger_full"), {}).get("viol", [])
        if a1 and bb:
            boot[f"{tier}: violation A1 vs B"] = paired_bootstrap_diff(a1, bb, seed=0)
    dim_report = {}
    for (tier, code, dim), c in dimcnt.items():
        n = sum(c.values()) or 1
        dim_report.setdefault(tier, {}).setdefault(code, {})[dim] = {
            "violated_rate": round(c.get("violated", 0) / n, 3), "n": n, "counts": dict(c)}

    stats = {"report": report, "bootstrap_violation": boot, "dim_report": dim_report}
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {STATS.relative_to(ROOT)}", flush=True)
    return stats


# ---------------------------------------------------------------------------
# rendering (shared by both modes)
# ---------------------------------------------------------------------------
def render(stats: dict) -> None:
    report = stats["report"]
    dim_report = stats.get("dim_report", {})

    for tier in TIERS:
        print(f"\n=== E6 persona robustness [{tier}] ===")
        td = canon_keys(report.get(tier, {}))
        for code in CODE:
            v = td.get(code)
            if not v:
                continue
            ov, oo = v["overall_violation"], v["ooc_rate"]
            print(f"  {PAPER[code]:<4} violation={ov['mean']:.3f}+/-{ov['sd']:.3f}  "
                  f"ooc={oo['mean']:.3f}+/-{oo['sd']:.3f}  break-turn={v['mean_break_turn']:.2f}")

    print("\n=== paired bootstrap: violation A1 - B (per tier; >0 -> A1 breaks more) ===")
    for tier in TIERS:
        td = canon_keys(report.get(tier, {}))
        a1 = td.get("bare", {}).get("overall_violation", {}).get("per_seed")
        b = td.get("linger_full", {}).get("overall_violation", {}).get("per_seed")
        if a1 and b:
            lo, hi = paired_bootstrap_diff(a1, b, seed=0)
            sig = "*" if (lo > 0 or hi < 0) else "(spans 0: no measurable gap — both rarely break)"
            print(f"  {tier:<5} A1 - B  [{lo}, {hi}]  {sig}")

    for tier in TIERS:
        print(f"\n=== per-dimension violation rate [{tier}] ===")
        print("  " + "dimension".ljust(24) + "".join(PAPER[c].rjust(7) for c in CODE))
        td = canon_keys(dim_report.get(tier, {}))
        for dim in DIMS:
            cells = [f"{td.get(c, {}).get(dim, {}).get('violated_rate'):.2f}"
                     if td.get(c, {}).get(dim) else "-" for c in CODE]
            print("  " + dim.ljust(24) + "".join(cc.rjust(7) for cc in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description="E6 persona-consistency analysis")
    ap.add_argument("--recompute", action="store_true",
                    help="re-judge raw runs into stats.json, then render (needs API key)")
    args = ap.parse_args()

    if args.recompute:
        stats = recompute_stats()
    else:
        if not STATS.exists():
            raise SystemExit(f"missing {STATS}\n  -> run `make e6` then `... analyze --recompute`, "
                             "or restore the cached file")
        stats = json.loads(STATS.read_text(encoding="utf-8"))
    render(stats)


if __name__ == "__main__":
    main()
