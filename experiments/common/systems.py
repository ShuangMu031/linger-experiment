"""The four-system factory: bare (A1) / rag (A2) / linger_no_factmem (B-) / linger_full (B).

All four share ONE backbone (the top tier read from .env) and receive the same per-seed
input stream; the only variable is architecture:

  bare              A1  bare LLM + full persona system prompt + sliding-window history.
  rag               A2  bare + BM25 retrieval over past user messages (top-3 injected).
  linger_no_factmem B-  full Linger orchestration, enable_user_fact_memory = False.
  linger_full       B   full Linger orchestration, enable_user_fact_memory = True.

build_llm_config() loads linger-bench/.env and wires the shared backbone. The cross-family
judges live in experiments.common.judges (kept separate so the dialogue layer never imports
the judge model). `python -m experiments.common.systems --smoke` runs ~10 live calls to
verify every system + the judge wire up; this is `make smoke`.
"""
from __future__ import annotations

import os

# Must be set BEFORE any linger_core import (process-level; load_dotenv(override=False) won't
# clobber it). B's long prompt is slow; a tight timeout silently triggers the mock fallback.
os.environ.setdefault("SF_TIMEOUT_SECONDS", "120")

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from experiments.common.naming import CODE
from experiments.common.persona import PERSONA_BRIEF
from experiments.common.retrieval import BM25Retriever

# parents[2]: experiments/common/systems.py -> [2] = repo root (where .env lives).
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# A turn callable takes the user text and returns (reply, used_real_backbone).
TurnFn = Callable[[str], Tuple[str, bool]]
SYSTEMS = CODE  # canonical four-system codes (single source: experiments.common.naming)


# ----------------------------------------------------------------------------
# Shared backbone config (reads .env; identical for all four systems)
# ----------------------------------------------------------------------------
def build_llm_config(timeout_seconds: int = 120):
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # override=False: does not clobber the SF_TIMEOUT_SECONDS set above

    from linger_core.config.default_config import DefaultConfig
    from linger_core.config.env_binding import (
        LLM_BINDINGS,
        apply_bindings,
        apply_special_overrides,
    )

    cfg = DefaultConfig()
    apply_bindings(cfg, LLM_BINDINGS)
    apply_special_overrides(cfg)
    for attr_path in (("llm", "top"), ("llm",)):  # best-effort: also lengthen the top-tier timeout
        obj = cfg
        try:
            for a in attr_path:
                obj = getattr(obj, a)
            if hasattr(obj, "timeout_seconds"):
                obj.timeout_seconds = timeout_seconds
        except Exception:  # noqa: BLE001
            pass
    return cfg


def resp_is_real(snap) -> bool:
    """Health check: did THIS turn's response_generation actually hit the real backbone
    (not the mock fallback)? Turns that fell back to mock are excluded from metrics."""
    for c in snap.llm_calls():
        if c.get("task_type") == "response_generation":
            return c.get("model") != "mock-llm" and not c.get("fallback_used")
    return False


# ----------------------------------------------------------------------------
# B / B- : the full Linger orchestration (single deterministic pipeline, not multi-agent)
# ----------------------------------------------------------------------------
@dataclass
class LingerSession:
    startup: Any
    turns: int = 0
    mock_turns: int = 0

    def turn(self, text: str) -> Tuple[str, Any, bool]:
        res = self.startup.foreground.handle_user_input(text)
        snap = self.startup.trace_archive.load_recent(count=1)[-1]
        used_real = resp_is_real(snap)
        self.turns += 1
        if not used_real:
            self.mock_turns += 1
        return res.get("message"), snap, used_real


def new_linger_session(warmup: bool = True, enable_user_fact_memory: bool = False) -> LingerSession:
    from linger_core.runtime.bootstrap.startup import Startup
    from linger_core.config.default_config import DefaultConfig

    cfg = DefaultConfig()
    cfg.memory.enable_user_fact_memory = enable_user_fact_memory  # the single flag isolating B vs B-
    s = Startup(load_memory=False, config=cfg)
    s.initialize()
    sess = LingerSession(s)
    if warmup:
        try:
            sess.turn("在吗")  # discarded: warm the connection + absorb first-turn cold start
        except Exception:  # noqa: BLE001
            pass
        sess.turns = 0
        sess.mock_turns = 0
    return sess


# ----------------------------------------------------------------------------
# A1 : bare LLM (same backbone) + full persona system prompt + sliding-window history.
#      The window truncation IS the bare LLM's intrinsic limit (forgets past the window) —
#      exactly the failure E5 measures.
# ----------------------------------------------------------------------------
@dataclass
class BareSession:
    provider: Any
    system_prompt: str
    history_window: int = 20  # keep the last N exchanges (2N messages)
    max_tokens: int = 400
    temperature: float = 0.7
    history: List[Any] = field(default_factory=list)
    turns: int = 0
    truncated_from: Optional[int] = None

    def turn(self, text: str) -> Tuple[str, bool]:
        from linger_core.llm.base import LLMMessage, LLMRequest
        from linger_core.llm.task_type import LLMTaskType

        keep = self.history_window * 2
        if len(self.history) > keep and self.truncated_from is None:
            self.truncated_from = self.turns
        msgs = (
            [LLMMessage(role="system", content=self.system_prompt)]
            + self.history[-keep:]
            + [LLMMessage(role="user", content=text)]
        )
        req = LLMRequest(messages=msgs, temperature=self.temperature, max_tokens=self.max_tokens)
        resp = self.provider.generate_for_task(LLMTaskType.RESPONSE_GENERATION, req)
        reply = resp.content if resp.success else ""
        used_real = bool(resp.success) and getattr(resp, "model", None) != "mock-llm"
        self.history.append(LLMMessage(role="user", content=text))
        self.history.append(LLMMessage(role="assistant", content=reply))
        self.turns += 1
        return reply, used_real


def new_bare_session(cfg, system_prompt: str = PERSONA_BRIEF, warmup: bool = True, **kw) -> BareSession:
    from linger_core.llm.tiered_provider import TieredProvider

    p = TieredProvider(cfg.llm)
    p.initialize()
    sess = BareSession(p, system_prompt, **kw)
    if warmup:
        try:
            sess.turn("在吗")
        except Exception:  # noqa: BLE001
            pass
        sess.history.clear()
        sess.turns = 0
        sess.truncated_from = None
    return sess


# ----------------------------------------------------------------------------
# A2 : A1 + BM25 retrieval over past user messages (the honest "what if bare just had a
#      memory lookup?" control). Reuses the dependency-free retriever in retrieval.py.
# ----------------------------------------------------------------------------
@dataclass
class RagSession:
    provider: Any
    system_prompt: str
    history_window: int = 20
    max_tokens: int = 300
    temperature: float = 0.7
    history: List[Any] = field(default_factory=list)
    retriever: BM25Retriever = field(default_factory=BM25Retriever)
    turns: int = 0

    def turn(self, text: str) -> Tuple[str, bool]:
        from linger_core.llm.base import LLMMessage, LLMRequest
        from linger_core.llm.task_type import LLMTaskType

        hits = self.retriever.search(text, top_k=3)
        mem = ("（你记得对方以前说过：" + "；".join(hits) + "）\n") if hits else ""
        keep = self.history_window * 2
        msgs = (
            [LLMMessage(role="system", content=self.system_prompt + "\n" + mem)]
            + self.history[-keep:]
            + [LLMMessage(role="user", content=text)]
        )
        req = LLMRequest(messages=msgs, temperature=self.temperature, max_tokens=self.max_tokens)
        resp = self.provider.generate_for_task(LLMTaskType.RESPONSE_GENERATION, req)
        reply = resp.content if getattr(resp, "success", False) else ""
        used_real = bool(getattr(resp, "success", False)) and getattr(resp, "model", None) != "mock-llm"
        self.retriever.add(text)  # remember what the user said this turn (for future retrieval)
        self.history.append(LLMMessage(role="user", content=text))
        self.history.append(LLMMessage(role="assistant", content=reply))
        self.turns += 1
        return reply, used_real


def new_rag_session(cfg, system_prompt: str = PERSONA_BRIEF, warmup: bool = True, **kw) -> RagSession:
    from linger_core.llm.tiered_provider import TieredProvider

    p = TieredProvider(cfg.llm)
    p.initialize()
    sess = RagSession(p, system_prompt, **kw)
    if warmup:
        try:
            sess.turn("在吗")
        except Exception:  # noqa: BLE001
            pass
        sess.history.clear()
        sess.retriever = BM25Retriever()
        sess.turns = 0
    return sess


# ----------------------------------------------------------------------------
# Unified factory: code name -> a turn callable (reply, used_real). Used by the run loops.
# ----------------------------------------------------------------------------
def make_turn_fn(system: str, cfg, history_window: int = 20, max_tokens: int = 300,
                 warmup: bool = True) -> TurnFn:
    if system == "bare":
        return new_bare_session(cfg, warmup=warmup, history_window=history_window,
                                max_tokens=max_tokens).turn
    if system == "rag":
        return new_rag_session(cfg, warmup=warmup, history_window=history_window,
                               max_tokens=max_tokens).turn
    if system in ("linger_no_factmem", "linger_full"):
        sess = new_linger_session(warmup=warmup, enable_user_fact_memory=(system == "linger_full"))

        def do(text: str) -> Tuple[str, bool]:
            reply, _snap, real = sess.turn(text)
            return reply, real

        return do
    raise ValueError(f"unknown system: {system!r} (expected one of {SYSTEMS})")


# ----------------------------------------------------------------------------
# Smoke self-test: B 3 turns + A1 3 turns + 1 judge call (~10 live calls). `make smoke`.
# ----------------------------------------------------------------------------
def _smoke() -> None:
    from experiments.common.judges import judge_persona_consistency

    print("[systems smoke] building config ...")
    cfg = build_llm_config()

    probe = ["最近上班好累，密室带团带到嗓子都哑了",
             "诶对了，上次跟你说的那个面试我过了！",
             "你还记得我前面说我在干啥工作不？"]

    print("\n### B (linger_full) — warmup + 3 turns ###")
    b = new_linger_session(warmup=True, enable_user_fact_memory=True)
    b_lines = []
    for t in probe:
        reply, _snap, real = b.turn(t)
        print(f"[USER] {t}")
        print(f"[鹿溪] {reply}  (real={real})")
        b_lines.append(f"用户: {t}\n鹿溪: {reply}")
    print(f"  B mock_turns={b.mock_turns}/{b.turns}")

    print("\n### A1 (bare) — warmup + 3 turns ###")
    a = new_bare_session(cfg, warmup=True, history_window=20, max_tokens=300)
    for t in probe:
        reply, real = a.turn(t)
        print(f"[USER] {t}")
        print(f"[A1]   {reply}  (real={real})")

    print("\n### Judge (Qwen2.5-72B, cross-family) ###")
    score, info = judge_persona_consistency(cfg, "\n".join(b_lines))
    print(f"  persona_consistency_score={score}")
    print(f"  {info}")
    print("\nSMOKE DONE")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="four-system factory; --smoke runs ~10 live calls")
    ap.add_argument("--smoke", action="store_true", help="verify all four systems + judge wire up")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    else:
        ap.print_help()
