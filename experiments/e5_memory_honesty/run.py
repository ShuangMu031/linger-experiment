"""E5 dialogue loop: four systems, high-density probes (default 160 turns), byte-identical input.

This stage only runs dialogue and collects raw rows (each recall's user/reply + the planted
fact index, rule-scored identity/boundary, a persona curve, and a mock health check). It does
NOT call the memory three-way judge — that (expensive=dialogue vs cheap=judging) is decoupled
into the analyze step, so a crashed run never wastes judging and a rubric change never re-runs
dialogue.

Each (system, seed) is checkpointed: it is written atomically the moment it finishes, and
re-running the same command skips already-completed runs (resume after a crash / network drop).

Run (one system per process, isolated trace dirs — see the Makefile `e5` target):
  THE_NEW_WORLD_DATA_DIR=data/e5/bare python -m experiments.e5_memory_honesty.run \
      --systems bare --out results/e5/bare.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SF_TIMEOUT_SECONDS", "120")  # before any linger_core import (via systems)

import argparse
import json
import time
from pathlib import Path

from experiments.common.systems import build_llm_config, make_turn_fn
from experiments.common.judges import judge_persona_consistency
from experiments.common.naming import CODE as SYSTEMS
from experiments.e5_memory_honesty.probes import (
    build_script,
    judge_boundary_violated,
    judge_identity_kept,
)


def run_system(system, script, cfg, judge_every=15, history_window=20, max_tokens=300):
    turn = make_turn_fn(system, cfg, history_window=history_window, max_tokens=max_tokens)
    rows, transcript, judge_curve = [], [], []
    n_id = n_id_kept = n_bd = n_bd_viol = mock = 0
    speaker = "鹿溪" if system.startswith("linger") else system

    for it in script:
        reply, real = turn(it.text)
        if not real:
            mock += 1
        transcript.append(f"用户: {it.text}\n{speaker}: {reply}")
        rec = {"turn": it.turn, "kind": it.kind, "user": it.text, "reply": reply, "real": real}

        if it.kind == "recall":
            rec["fact_idx"] = it.meta.get("fact_idx")
            rec["key"] = it.meta.get("key", [])
            # memory three-way is decided by the judge in analyze; not scored here
        elif it.kind == "identity":
            kept = judge_identity_kept(reply, it)
            rec["kept"] = kept
            n_id += 1
            n_id_kept += int(kept)
        elif it.kind == "boundary":
            viol = judge_boundary_violated(reply, it)
            rec["violated"] = viol
            n_bd += 1
            n_bd_viol += int(viol)
        rows.append(rec)

        if it.turn > 0 and it.turn % judge_every == 0:
            sc, _ = judge_persona_consistency(cfg, "\n".join(transcript[-6:]))
            if sc is not None:
                judge_curve.append((it.turn, sc))

    metrics = {
        "identity_retention": round(n_id_kept / n_id, 3) if n_id else None,
        "boundary_violation_rate": round(n_bd_viol / n_bd, 3) if n_bd else None,
        "judge_mean": round(sum(s for _, s in judge_curve) / len(judge_curve), 1) if judge_curve else None,
        "mock_turns": mock, "total_turns": len(script),
        "n_recall": sum(1 for r in rows if r["kind"] == "recall"),
    }
    return {"metrics": metrics, "judge_curve": judge_curve, "rows": rows}


def _load_done(out_path: Path):
    """Read already-finished runs; return (accumulated dict, set of completed keys)."""
    if out_path.exists():
        d = json.loads(out_path.read_text(encoding="utf-8"))
        return d, set(d.get("runs", {}).keys())
    return {"setup": {}, "runs": {}}, set()


def _atomic_write(out_path: Path, obj):
    """Write to a temp file then rename, so an interrupted run never leaves half-written JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=160)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    ap.add_argument("--judge-every", type=int, default=15)
    ap.add_argument("--history-window", type=int, default=20)
    ap.add_argument("--out", default="results/e5/e5.json")
    args = ap.parse_args()

    cfg = build_llm_config()
    out_path = Path(args.out)  # relative to cwd (the repo root, per the Makefile)
    out, done = _load_done(out_path)  # seed-level resume: completed (system,seed) keys are skipped
    out["setup"] = {"turns": args.turns, "seeds": args.seeds, "systems": args.systems,
                    "judge_every": args.judge_every, "history_window": args.history_window,
                    "backbone": "deepseek-ai/DeepSeek-V4-Flash",
                    "note": "simulation turns != calendar time; bare/rag/linger_* share one backbone; "
                            "memory three-way + stats are in analyze"}
    t0 = time.time()
    for seed in args.seeds:
        script = build_script(n_turns=args.turns, seed=seed, history_window=args.history_window)
        for system in args.systems:
            key = f"{system}_seed{seed}"
            if key in done:
                print(f"[{key}] skip (checkpointed)", flush=True)
                continue
            print(f"[{key}] running {args.turns} turns ...", flush=True)
            r = run_system(system, script, cfg, args.judge_every, args.history_window)
            out["runs"][key] = r
            _atomic_write(out_path, out)  # checkpoint after every seed -> crash/resume safe
            m = r["metrics"]
            print(f"  -> id={m['identity_retention']} bd={m['boundary_violation_rate']} "
                  f"judge={m['judge_mean']} mock={m['mock_turns']}/{m['total_turns']} "
                  f"n_recall={m['n_recall']} ({time.time()-t0:.0f}s) saved", flush=True)

    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()
