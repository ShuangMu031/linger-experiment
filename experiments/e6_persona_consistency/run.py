"""E6 persona-consistency dialogue loop: four systems x seed x 12 scripts, 5 turns each.

Each script runs in a FRESH session (no cross-script context). This stage collects raw rows
(per turn user/reply/real); the held/violated/ooc/break-turn judging is decoupled into the
analyze step (analyze --recompute), exactly like E5. Seed-level checkpoint/resume.

Run (one system per process, isolated trace dirs — see the Makefile `e6` target, or all four
at once with the default --systems):
  THE_NEW_WORLD_DATA_DIR=data/e6 python -m experiments.e6_persona_consistency.run \
      --out results/e6/persona.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SF_TIMEOUT_SECONDS", "120")  # before any linger_core import (via systems)

import argparse
import json
import time
from pathlib import Path

from experiments.common.naming import CODE as SYSTEMS
from experiments.common.systems import build_llm_config, make_turn_fn
from experiments.e6_persona_consistency.scripts import PERSONA_SCRIPTS


def run_script(system, script, cfg):
    """Run one 5-turn script in a fresh session; return rows (per turn user/reply/real)."""
    turn = make_turn_fn(system, cfg)  # fresh session per script
    rows = []
    mock = 0
    for ti, text in enumerate(script["turns"]):
        reply, real = turn(text)
        if not real:
            mock += 1
        rows.append({"turn": ti, "user": text, "reply": reply, "real": real})
    return {"id": script["id"], "tier": script["tier"], "dim": script["dim"],
            "goal": script["goal"], "rows": rows, "mock_turns": mock}


def _load_done(out_path: Path):
    if out_path.exists():
        d = json.loads(out_path.read_text(encoding="utf-8"))
        return d, set(d.get("runs", {}).keys())
    return {"setup": {}, "runs": {}}, set()


def _atomic_write(out_path: Path, obj):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    ap.add_argument("--out", default="results/e6/persona.json")
    args = ap.parse_args()

    cfg = build_llm_config()
    out_path = Path(args.out)  # relative to cwd (the repo root, per the Makefile)
    out, done = _load_done(out_path)
    out["setup"] = {"seeds": args.seeds, "systems": args.systems, "n_scripts": len(PERSONA_SCRIPTS),
                    "backbone": "deepseek-ai/DeepSeek-V4-Flash",
                    "note": "fresh session per script; simulation turns != calendar time; "
                            "held/violated/ooc judging is in analyze"}
    t0 = time.time()
    for seed in args.seeds:
        for system in args.systems:
            key = f"{system}_seed{seed}"
            if key in done:
                print(f"[{key}] skip (checkpointed)", flush=True)
                continue
            print(f"[{key}] running {len(PERSONA_SCRIPTS)} scripts ...", flush=True)
            scripts_out = [run_script(system, s, cfg) for s in PERSONA_SCRIPTS]
            out["runs"][key] = {"scripts": scripts_out}
            _atomic_write(out_path, out)
            nmock = sum(s["mock_turns"] for s in scripts_out)
            print(f"  -> {len(scripts_out)} scripts, mock_turns={nmock} ({time.time()-t0:.0f}s) saved",
                  flush=True)
    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()
