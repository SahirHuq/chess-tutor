# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A terminal chess tutor: you load one of your games (PGN) and talk to it. A Gemini
model explains moves in plain English but **cannot reason about the board itself** —
it only narrates facts a real engine (Stockfish + python-chess) hands back through
tools. That anti-hallucination contract is the whole point of the design; every
design decision serves it.

## Commands

```bash
# Run (needs a free key from aistudio.google.com)
export GEMINI_API_KEY=...
.venv/bin/python tutor.py              # interactive: paste / file path / 'sample'
.venv/bin/python tutor.py mygame.pgn   # load a .pgn file
pbpaste | .venv/bin/python tutor.py -  # pipe a PGN from stdin

# Tests — pytest is the ONLY check configured (no linter/typechecker in the repo)
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest test_compare.py -q                    # one file
.venv/bin/python -m pytest test_compare.py::test_compare_verdict_flags_forced_mate_gap   # one test
```

Always use `.venv/bin/python`, never system Python. **Stockfish is a hard runtime
dependency**, found via `$STOCKFISH_PATH` or `PATH` (`brew install stockfish`).
Engine-backed tests skip themselves when it's absent (the `needs_engine` marker in
`test_compare.py`); pure-logic tests always run.

## Architecture: the "voice" vs the "brain"

```
you ─▶ Gemini 2.5 Flash-Lite ──tools──▶ python-chess (rules) + Stockfish (eval)
         (voice: tutor.py)                (brain / source of truth)
```

- **Brain (deterministic, no model):** `engine.py` (Stockfish → scored candidate
  lines), `features.py` (pure, engine-free position facts), `session.py` (game
  state + safe what-if branching). Plain and unit-testable.
- **Voice (the model):** `tutor.py` holds the system prompt and the only
  provider-aware code. Forbidden by prompt to state any move, eval, or line a tool
  didn't return this turn.
- **Bridge:** `tools.py` is the *only* surface the model can touch. Legality and
  truth are enforced here, never by the model.

## Invariants — violate these and bugs are silent

- **Evals are always White's POV.** Positive = White better, negative = Black
  better. A forced mate is `±100000` centipawns (`_MATE_VALUE`). Every comparison
  re-signs to the *mover's* POV explicitly (`_to_mover_value`, `_classify_move`,
  `_position_value_white`). Wrong sign → verdicts silently invert.
- **The mainline is read-only.** `session.current` is a movable cursor; `goto_move`
  navigates the real game, `play_move` branches off a *copy* onto `branch_stack`,
  `go_back` pops it. The real game is never mutated — tests assert this.
- **Grounding-only output.** Tools return engine/python-chess-derived facts, never
  adjectives. The model can only be as honest as the data you hand it: return
  concrete captures, net material, and *named* positional changes — not "activity"
  or "pressure." The prompt enforces "if a tool didn't return it, don't say it."
- **SAN before push.** A move's SAN must be computed on the board *before* the move
  is pushed (see `features.annotate_line`). Pushing first corrupts the notation.

## Data shapes the tools pass around

Knowing these lets you write new tools/features without re-reading every file.

```python
# engine.Analysis.to_dict()
{"fen", "side_to_move", "depth", "outcome",   # outcome set only for game-over
 "best_moves": [ {                            # ranked, len == multipv (default 3)
     "rank", "move",            # SAN, e.g. "Nf3"
     "eval",                    # human string, White POV: "+0.42 (white better)"
     "line",                    # principal variation in SAN
     "white_centipawns",        # int or None (None on forced mate)
     "mate_in",                 # signed mate-in-N, White POV (+white/-black), or None
 } ] }

# features.position_facts(board) — the grounding facts
{"fen","side_to_move","fullmove_number","in_check","checkers",
 "material": {"white","black","diff"},        # diff = White - Black
 "hanging_pieces": [...], "castling_rights": {...}}

# features.annotate_line(board, moves, max_plies) — the "mechanism" of a line
{"summary",                                   # a ready-to-relay factual sentence
 "moves": [ {"move","side","from","captures","attacks","gives_check"} ],
 "material_before","material_after","material_swing_white_minus_black"}

# features.positional_changes(before, after) -> list[str]
# e.g. ["white gave up the bishop pair", "black now has doubled pawns on the b-file"]

# tools._evaluate_candidate(...) — the shared per-move verdict core
{"move","verdict","centipawns_lost","played_engine_best","move_does",
 "consequence","position_changes","engine_after","facts_after","value_white_after"}
```

Engine defaults (`engine.py`): depth 18, multipv 3, PV shown 8 plies.
Verdict bands by centipawns lost (`_classify_move`): ≤15 best · ≤50 good ·
≤120 inaccuracy · ≤300 mistake · else blunder.

## How code is written here (match this style)

- **Literate docstrings that explain *why*, not what.** Every module opens with a
  one-line metaphor for its role ("the deterministic *judgement* half of the
  brain"). Functions justify their design and grounding rationale, not their
  mechanics. New code should read the same — terse where obvious, expansive where a
  decision needs defending.
- **Comments explain reasoning and edge cases**, never restate the code.
- `from __future__ import annotations`, explicit type hints, named exports,
  snake_case, pure functions in `features.py`.
- **Error results, not exceptions, at the tool/session boundary:**
  `{"ok": False, "error": "..."}`. Exceptions are for truly exceptional cases.
- Prefer the *specific* named concept the engine/features produced (e.g. "you gave
  up the bishop pair") over generic advice ("be careful trading"). Honesty over
  fabrication is the repo's governing value.

## The tool layer (`tools.py`)

Bound once per run via `tools.bind(session, engine)` (sets module global `_CTX`);
tools reach state through `_ctx()`. `ALL_TOOLS` is the exact list handed to
`google-genai` as function schemas — **the docstring is the schema the model
reads**, so write tool docstrings for the model: when to call, what comes back, how
to present it. Every tool calls `_log(...)` so each engine consult prints dimmed as
`[engine tool] ...` (the grounding, made visible).

`explain_move` (a played move) and `compare_moves` (two hypothetical moves from the
current position) share **`_evaluate_candidate`**, which operates on `chess.Board`
objects only — no session coupling. Change a move's analysis there, not in two
places. `_recommended_plan` (the engine's own best line) is likewise shared.
`explain_move`'s return-dict keys are referenced verbatim by the system prompt in
`tutor.py` — preserve them when refactoring.

## The model loop (`tutor.py`)

`GeminiTutor` drives a **manual** function-calling loop (SDK auto-calling disabled)
so every call is logged, tool errors become structured results instead of crashes,
transient 5xx are retried, and an empty post-tool answer gets one nudge. This is the
*single* module that knows the provider — reimplement `GeminiTutor` to swap models;
nothing else mentions Gemini. The model is small (`gemini-2.5-flash-lite`), so a new
tool's prompt clause must be explicit about *when* to call it and *how* to structure
the answer, or it drops steps.

## Adding a new tool — checklist

1. Implement in `tools.py`, returning grounded facts; call `_log(...)`. Reuse
   `_evaluate_candidate` / `features.*` rather than recomputing geometry.
2. Add to `ALL_TOOLS`.
3. Add a routing clause to `SYSTEM_TEMPLATE` in `tutor.py` (when + how to present).
4. Enforce legality via `session.parse_move` / `session` — never trust the model.
5. Test: pure-logic helpers without the engine, plus an engine-backed flow under the
   `needs_engine` skip guard.

## Known rough edge

When the engine's best move is a *forced mate*, the alternative's `centipawns_lost`
becomes a huge number (mate and centipawns share one scale). `verdict` and
`comparison.note` convey it correctly; the raw centipawn figure does not.
