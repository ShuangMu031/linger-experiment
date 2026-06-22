# Linger-Bench entry points.
#
# Track 1 (offline, no API key):  make figures
# Track 2 (end-to-end, needs .env + vendored linger_core):  make smoke / e5 / e6
#
# E5 runs each system in its own process with an isolated trace dir so the B/B-
# trace archives never collide.

PY ?= python
DATA ?= data

.PHONY: help smoke e5 e6 figures analyze test clean

help:
	@echo "make smoke    - ~10 LLM calls, verify the four systems wire up (needs .env)"
	@echo "make e5       - full E5 memory-honesty run (four systems, hours, costs money)"
	@echo "make e6       - full E6 persona-consistency run"
	@echo "make figures  - replot from cached results/ (offline, no API key)"
	@echo "make analyze  - re-run judges/statistics on cached dialogues"
	@echo "make test     - pure-logic self-tests (no LLM)"

smoke:
	$(PY) -m experiments.common.systems --smoke

e5:
	@echo "Running four systems in parallel with isolated trace dirs..."
	THE_NEW_WORLD_DATA_DIR=$(DATA)/e5/bare  $(PY) -m experiments.e5_memory_honesty.run --systems bare              --out results/e5/bare.json &
	THE_NEW_WORLD_DATA_DIR=$(DATA)/e5/rag   $(PY) -m experiments.e5_memory_honesty.run --systems rag               --out results/e5/rag.json &
	THE_NEW_WORLD_DATA_DIR=$(DATA)/e5/bmin  $(PY) -m experiments.e5_memory_honesty.run --systems linger_no_factmem --out results/e5/linger_no_factmem.json &
	THE_NEW_WORLD_DATA_DIR=$(DATA)/e5/bful  $(PY) -m experiments.e5_memory_honesty.run --systems linger_full       --out results/e5/linger_full.json &
	wait

e6:
	$(PY) -m experiments.e6_persona_consistency.run --out results/e6/persona.json

analyze:
	$(PY) -m experiments.e5_memory_honesty.analyze
	$(PY) -m experiments.e6_persona_consistency.analyze

figures:
	$(PY) -m figures.plot_memory_honesty

test:
	$(PY) -m experiments.e5_memory_honesty.probes      # script-shape + rule-scoring selftest
	$(PY) -m experiments.common.stats                  # statistics selftest
	$(PY) -m experiments.e6_persona_consistency.scripts

clean:
	rm -rf $(DATA)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
