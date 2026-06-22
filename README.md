# Linger-Bench

Reproduction code for **"Decision–Expression Separation for Honest Memory and
Persona Governance in Long-Term LLM Companions."**

Linger-Bench evaluates a long-term LLM companion ("Linger") against bare-LLM and
retrieval-augmented baselines on two failure modes that matter for companions
people talk to for weeks:

1. **Honest memory** — when asked about a fact the user mentioned long ago, does
   the system recall it correctly, *hallucinate* a confident-but-wrong fact, or
   *honestly abstain*? (Experiment E5)
2. **Persona governance** — under escalating pressure (rename, romance coercion,
   jailbreak, gaslighting, forced "I am just an AI"), does the character hold its
   identity and boundaries? (Experiment E6)

---

## ⚠️ Research-integrity notice (read first)

This repository is deliberately explicit about what is and is not a real
end-to-end measurement. Please preserve these statements if you fork it.

- **E1–E4 use *synthetic* data.** They are sanity checks of the metric
  definitions and pipeline plumbing, **not** real deployment logs and **not** a
  real 30-day trial run. They are kept for completeness and clearly labelled as
  synthetic everywhere they appear. See [`experiments/synthetic_sanity_checks/`](experiments/synthetic_sanity_checks/).
- **E5 and E6 are real, end-to-end LLM runs.** Every dialogue turn is a live API
  call to the backbone model; turns that silently fell back to a mock provider
  are detected and excluded (`resp_is_real` health-check). Judges are a
  **cross-family** model (Qwen2.5-72B) scoring at `temperature=0`.
- **The system under test is rule-based cognitive orchestration, not a
  multi-agent system.** A single deterministic Python pipeline calls the LLM for
  a few well-scoped sub-steps (interpret / decide / generate / extract). There
  are no autonomous agents messaging each other. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
- **"Turns" are simulation turns, not calendar time.** Long-range memory is
  stressed by placing the recall probe far beyond the sliding-context window, not
  by waiting real days.

---

## The four systems

All four systems share the **same backbone model** and receive the **same
byte-for-byte input stream** (fixed per seed). The only variable is architecture.

| Paper name | Code name            | What it is                                                                 |
|------------|----------------------|---------------------------------------------------------------------------|
| **A1**     | `bare`               | Bare LLM + full persona system prompt + sliding-window history. Forgets anything beyond the window. |
| **A2**     | `rag`                | A1 + a simple BM25 retriever over past user messages (top-3 injected).     |
| **B⁻**     | `linger_no_factmem`  | Full Linger orchestration with `enable_user_fact_memory = False` (status quo, fact-blind). |
| **B**      | `linger_full`        | Full Linger orchestration with `enable_user_fact_memory = True` (extract → store → retrieve → inject user facts). |

The B vs B⁻ contrast isolates exactly one capability: governed *user-fact*
memory. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the flag's effect.

---

## Repository layout

```
linger-bench/
├── README.md
├── LICENSE
├── pyproject.toml              # package + dependency declaration
├── requirements.txt
├── .env.example                # API key / backbone model template
├── .gitignore                  # ignores data/ (raw traces) and .env
├── Makefile                    # one-line entry points
│
├── linger_core/                # ← vendored backend (the B system). NOT included;
│                               #    see "Vendoring linger_core" below.
│
├── experiments/
│   ├── common/                 # shared infrastructure
│   │   ├── persona.py          # persona brief (system prompt + judge reference)
│   │   ├── systems.py          # the four-system factory (bare / rag / linger)
│   │   ├── judges.py           # cross-family judges (persona consistency + memory 3-way)
│   │   └── stats.py            # Wilson CI, two-proportion z, Holm, paired bootstrap
│   │
│   ├── e5_memory_honesty/
│   │   ├── probes.py           # 22 memory + 6 identity + 6 boundary probes; rule scoring
│   │   ├── run.py              # dialogue loop (checkpointed, resumable)
│   │   └── analyze.py          # memory three-way classification + significance tests
│   │
│   ├── e6_persona_consistency/
│   │   ├── scripts.py          # 12 escalating induction scripts (gentle + adversarial)
│   │   ├── run.py              # per-turn judging (identity / boundary / ooc)
│   │   └── analyze.py          # break-rate by dimension + OOC rate + bootstrap
│   │
│   └── synthetic_sanity_checks/  # E1–E4 (SYNTHETIC — see its README)
│       └── README.md
│
├── figures/
│   └── plot_memory_honesty.py  # E5 stacked-bar (correct / hallucinated / honest)
│
├── results/                    # cached result JSON (committed, ~MBs) — replots offline
│   ├── e5/  e6/  synthetic/
│
├── data/                       # raw per-turn traces (git-ignored; regenerated by run.py)
│
└── docs/
    ├── ARCHITECTURE.md         # the B-system backend: 7-stage pipeline, the flag, why not multi-agent
    ├── INTEGRITY.md            # the integrity statements above, expanded
    └── REPRODUCE.md            # step-by-step, both tracks
```

---

## Quickstart

### Track 1 — offline replot (zero cost, no API key)

Reproduce every figure/table in the paper from the committed result JSON:

```bash
pip install -e .
make figures            # reads results/e5/*.json → figures/*.pdf
python -m experiments.e5_memory_honesty.analyze --from-cache
python -m experiments.e6_persona_consistency.analyze
```

### Track 2 — end-to-end re-run (needs an API key + a vendored `linger_core`)

```bash
cp .env.example .env    # then fill in SILICONFLOW_API_KEY
make smoke              # ~10 LLM calls; verifies the four systems wire up
make e5                 # full E5 (four systems, 5 seeds × 160 turns) — hours, costs money
make e6                 # full E6 persona stress test
```

> A1/A2 (bare / rag) only need `linger_core.llm.*` (the provider layer). B/B⁻
> need the full vendored backend — see below.

---

## Vendoring `linger_core` (required only for the B system)

The B system *is* the companion backend; it cannot be reduced to a few files. To
re-run B/B⁻ end-to-end, vendor the backend package and rename it:

1. Copy the backend package into `linger-bench/linger_core/`.
2. Rewrite the import prefix `the_new_world` → `linger_core` (a global,
   mechanical rename).
3. Drop the non-experiment surface: `web/`, `cli/`.
4. Run the smoke test: `make smoke`.

If you only want to reproduce the bare/RAG baselines or replot cached results,
you can skip vendoring entirely.

---

## Reproducing each experiment

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for exact commands, expected
runtimes, checkpoint/resume behaviour, and the cached-vs-live decision per
experiment.

---

## Citation

```bibtex
@article{linger_decision_expression,
  title  = {Decision--Expression Separation for Honest Memory and Persona
            Governance in Long-Term LLM Companions},
  author = {<authors>},
  year   = {2026},
  note   = {Code: https://github.com/<you>/linger-bench}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
