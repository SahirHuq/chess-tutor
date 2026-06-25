# Conversational AI Chess Tutor — MVP

A terminal program where you load one of your games and **talk to it**. The AI
explains your moves in plain English, but it can't make up chess: it never
reasons about the board itself — it only narrates facts a real engine hands it.

```
  you ──▶ Gemini 2.5 Flash-Lite ──tools──▶ python-chess (rules) + Stockfish (eval)
            (the "voice")                       (the "brain" / source of truth)
```

The model reaches the engine only through tools (`goto_move`, `analyze`,
`play_move`, `go_back`). A hard rule — *"only discuss what a tool returned"* —
is what prevents hallucination. Your real game is read-only; every "what-if"
runs on a throwaway copy.

## Files
| File | Role |
|---|---|
| `engine.py` | Stockfish wrapper → scores + best lines |
| `features.py` | Position facts (material, hanging pieces, checks) — the grounding |
| `session.py` | Game state + safe what-if branching |
| `tools.py` | The tools the model is allowed to call |
| `tutor.py` | CLI + the Gemini adapter |
| `sample.pgn` | A test game with a known blunder (9.Nxd4) |
| `test_features.py`, `test_session.py` | Unit tests for the brain/state |

## Setup (already done on this machine)
- Stockfish: `brew install stockfish`
- Python deps in a project venv: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`

## Run
1. Get a free Gemini API key at <https://aistudio.google.com>, then:
   ```bash
   export GEMINI_API_KEY=...
   ```
2. Start it (three ways to load a game):
   ```bash
   .venv/bin/python tutor.py                     # interactive: choose paste / file / sample
   .venv/bin/python tutor.py mygame.pgn          # load a .pgn file
   pbpaste | .venv/bin/python tutor.py - --color white  # pipe a PGN in (macOS)
   ```
   In interactive mode, type `paste`, then paste your **Lichess or Chess.com PGN**
   export (headers, clocks, and result are all fine), and finish with a line
   containing just `END`.

   After loading, the tutor shows the two players and asks **which side you
   played** — so "what was my biggest mistake?" looks at *your* moves, not your
   opponent's. (It never guesses your colour from who won.) When piping a PGN via
   stdin there's no keyboard left to ask on, so pass `--color white|black`; omit it
   and player-specific questions will simply ask you which side you played.
3. Ask away — about your move quality, the idea behind the better move, what-ifs:
   - `was 12.Nf3 a good move, and what's the idea?`
   - `why did my eval drop on move 18? what should I have played?`
   - `what was my biggest mistake?` · `what did my opponent miss?` · `what was the turning point?`
   - `what if I had castled instead of h3?`
   - (on the sample game) `why is 9.Nxd4 a blunder?`

Each tool call is printed dimmed as `[engine tool] ...` so you can **see** the
model consulting the engine — that's the grounding, visible and auditable.

## Tests
```bash
.venv/bin/python -m pytest -q
```

## Cost
Realistically **$0** on Gemini's free tier for interactive testing.
