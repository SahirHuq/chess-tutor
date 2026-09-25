"""The tools the model is allowed to press — its only window onto the engine.

Each function is handed to `google-genai` as a callable tool. The model decides
when to call them; the SDK runs them and feeds the (real, engine-derived) result
back. Every tool returns the current position so the model stays oriented, and
logs its call so the grounding is auditable in the CLI.

Legality is enforced here (via `session`) — never by the model.
"""

from __future__ import annotations

import re
from typing import Optional

import chess

import features
import tactics
from engine import Analysis, Engine
from session import SessionState

_DIM = "\033[2m"
_RESET = "\033[0m"


class ToolContext:
    """Holds the live session + engine the tool functions act on."""

    def __init__(self, session: SessionState, engine: Engine) -> None:
        self.session = session
        self.engine = engine
        self.calls: list[str] = []


_CTX: Optional[ToolContext] = None


def bind(session: SessionState, engine: Engine) -> ToolContext:
    """Attach the live session + engine the tools operate on (call once per run)."""
    global _CTX
    _CTX = ToolContext(session, engine)
    return _CTX


def _ctx() -> ToolContext:
    if _CTX is None:
        raise RuntimeError("[tools] not bound — call tools.bind(session, engine) first")
    return _CTX


def _log(call: str) -> None:
    ctx = _ctx()
    ctx.calls.append(call)
    print(f"  {_DIM}[engine tool] {call}{_RESET}", flush=True)


def _eval_summary(analysis: dict) -> str:
    if analysis.get("outcome"):
        return analysis["outcome"]
    best = analysis.get("best_moves") or []
    if not best:
        return "(no moves)"
    top = best[0]
    return f"best {top['move']} {top['eval']}"


def _pos_summary(result: dict) -> str:
    if not result.get("ok", True):
        return result.get("error", "error")
    last = result.get("last_move") or "start"
    return (
        f"{last} | {result['side_to_move']} to move | "
        f"branch_depth={result['branch_depth']}"
    )


# ---- the tools the model can call ------------------------------------------


def goto_move(ply: int) -> dict:
    """Move the board to a specific point in the user's real game.

    `ply` is the half-move index from the move list in your instructions:
    ply 0 is the starting position; ply N is the position right AFTER the Nth
    half-move was played. Use this to position the board at the move the user is
    asking about (e.g. to discuss "15.Nxe5", go to the ply just before it to see
    the alternatives, or to its own ply to see the result). This leaves any
    what-if branch and returns to the real game.
    """
    result = _ctx().session.goto_move(ply)
    _log(f"goto_move({ply}) -> {_pos_summary(result)}")
    return result


def analyze() -> dict:
    """Ask the chess engine to evaluate the CURRENT position.

    Returns the engine's top candidate moves (each with an evaluation and the
    line that follows) plus grounding facts: material balance, hanging pieces,
    and any checks. You MUST call this before stating who is better, what the
    best move is, or why a move is good or bad — never rely on memory.
    """
    ctx = _ctx()
    board = ctx.session.current
    fen = board.fen()
    analysis = ctx.session.analysis_cache.get(fen)
    if analysis is None:
        analysis = ctx.engine.analyse(board).to_dict()
        ctx.session.analysis_cache[fen] = analysis
    _log(f"analyze() -> {_eval_summary(analysis)}")
    return {
        "position": ctx.session.orientation(),
        "engine": analysis,
        "facts": ctx.session.current_facts(),
    }


def explain_position() -> dict:
    """Get the COMPLETE grounded picture of the CURRENT position — use this for
    OPEN-ENDED, CONCEPTUAL questions that do NOT name a single move and are NOT about
    finding a mistake: "is my position better or worse, and why?", "what are the
    imbalances here?", "was I playing too passively?", "is my bishop good or bad?",
    "what should my plan be?", "what are the weaknesses in this position?". First call
    goto_move(ply) if the user means a specific point in the game.

    Returns, all for this one position:
      - `engine`: the evaluation and the top candidate moves with their lines (who is
        better, and the best plan) — use it for "who's better" and "what to play",
      - `facts`: material balance, hanging pieces, checks, castling rights,
      - `features`: the positional CHARACTER for BOTH sides — bishop pair; pawn
        structure (doubled / isolated / backward / connected / passed pawns); rooks on
        open files; knight outposts; trapped pieces; king safety; piece activity
        (`mobility`); development; and centre control.
    REASON over all of it and TEACH: connect the facts into an explanation the player
    can learn from (e.g. "you have the bishop pair and more space and your opponent has
    a backward d-pawn, so keep the position open and pressure that pawn"). You may draw
    conclusions the data supports even if no single field is named for them — BUT every
    CONCRETE claim (a specific move, an evaluation, a square, a piece, a capture) must
    come from this snapshot. Never invent a move or eval, and never describe a square or
    piece the data does not mention.
    """
    ctx = _ctx()
    board = ctx.session.current
    fen = board.fen()
    analysis = ctx.session.analysis_cache.get(fen)
    if analysis is None:
        analysis = ctx.engine.analyse(board).to_dict()
        ctx.session.analysis_cache[fen] = analysis
    _log(f"explain_position() -> {_eval_summary(analysis)}")
    return {
        "position": ctx.session.orientation(),
        "engine": analysis,
        "facts": ctx.session.current_facts(),
        "features": features.positional_features(board),
    }


def _normalize_san(text: str) -> str:
    """Strip move numbers and check/annotation marks so moves compare cleanly."""
    text = re.sub(r"^\s*\d+\.(\.\.)?\s*", "", text.strip())  # drop "9." / "9..."
    return text.rstrip("+#!?").strip()


def _resolve_ply(session: SessionState, move: str) -> Optional[int]:
    """Find the 1-based ply of a move written as the user refers to it.

    Honors a move number / side when given ("9.Nxd4" = White's 9th half-move,
    "9...Nxd4" = Black's 9th) so a SAN that occurs more than once in the game
    resolves to the RIGHT occurrence — not just the first one. Move N maps to
    ply 2N-1 (White) or ply 2N (Black). Falls back to the first SAN match only
    when no usable move number is given.
    """
    text = move.strip()
    target = _normalize_san(text)

    header = re.match(r"\s*(\d+)\s*(\.\.\.|\.)?", text)
    if header:
        move_no = int(header.group(1))
        marker = header.group(2)
        if marker == "...":
            candidates = [2 * move_no]  # Black's move
        elif marker == ".":
            candidates = [2 * move_no - 1]  # White's move
        else:
            candidates = [2 * move_no - 1, 2 * move_no]  # side unspecified
        for ply in candidates:
            idx = ply - 1
            if 0 <= idx < len(session.mainline_san) and _normalize_san(session.mainline_san[idx]) == target:
                return ply

    for i, san in enumerate(session.mainline_san):  # fallback: first SAN match
        if _normalize_san(san) == target:
            return i + 1
    return None


_MATE_VALUE = 100000  # stand-in centipawns for a forced mate, White's POV


def _white_value(best_moves: list[dict]) -> Optional[int]:
    """The best line's evaluation in centipawns from White's POV (mate -> ±100000)."""
    if not best_moves:
        return None
    top = best_moves[0]
    if top.get("mate_in") is not None:
        return _MATE_VALUE if top["mate_in"] > 0 else -_MATE_VALUE
    return top.get("white_centipawns")


def _position_value_white(board: chess.Board, analysis_dict: dict) -> Optional[int]:
    """White-POV value of `board`: the engine's best-line score, or ±100000 for a
    terminal checkmate and 0 for a terminal draw. A mating move leaves no candidate
    moves to score, so without this branch it would read as 'unclear'."""
    best = analysis_dict.get("best_moves")
    if best:
        return _white_value(best)
    if board.is_checkmate():  # the side to move is the one checkmated (i.e. losing)
        return -_MATE_VALUE if board.turn else _MATE_VALUE
    if board.is_game_over(claim_draw=True):
        return 0
    return None


def _classify_move(
    mover: str,
    best_value: Optional[int],
    actual_value: Optional[int],
    played_the_best: bool,
) -> dict:
    """Judge a move by how far it falls short of the engine's best, for the mover."""
    if played_the_best:
        return {"verdict": "best move", "centipawns_lost": 0}
    if best_value is None or actual_value is None:
        return {"verdict": "unclear", "centipawns_lost": None}
    loss = (best_value - actual_value) if mover == "white" else (actual_value - best_value)
    loss = max(int(loss), 0)  # you can't do better than "best"; tiny negatives are noise
    if loss <= 15:
        verdict = "best move"
    elif loss <= 50:
        verdict = "good move"
    elif loss <= 120:
        verdict = "inaccuracy"
    elif loss <= 300:
        verdict = "mistake"
    else:
        verdict = "blunder"
    return {"verdict": verdict, "centipawns_lost": loss}


def _recommended_plan(
    before_board: chess.Board, before_analysis: Analysis, best_list: list[dict]
) -> Optional[dict]:
    """The engine's own top move from `before_board`, the line it intends, and what
    that move achieves — the "what's the ideal here" material. Shared by
    explain_move and compare_moves so neither recomputes it."""
    if not before_analysis.lines or not best_list:
        return None
    top = before_analysis.lines[0]
    if not top.moves:
        return None
    plan_board = before_board.copy()
    plan_board.push(top.moves[0])
    return {
        "move": top.move,
        "eval": best_list[0]["eval"],
        # Each plan move annotated with exactly what it captures/attacks, so the
        # idea can be described precisely instead of guessed.
        "plan": features.annotate_line(before_board, top.moves, max_plies=6)["moves"],
        "achieves": features.positional_changes(before_board, plan_board),
    }


def _evaluate_candidate(
    engine: Engine, before_board: chess.Board, before_analysis: Analysis, move: chess.Move
) -> dict:
    """Everything about playing `move` from `before_board`: its verdict, how many
    centipawns it loses vs the engine's best, what it does, the consequence (the
    move + the opponent's best reply with the NET material result), and the
    positional delta. Operates on boards only — no session state — so explain_move
    (a played move) and compare_moves (two hypothetical moves) share one grounded
    analysis path. `before_analysis` is the engine's verdict on `before_board`, so
    the best-move reference is computed once and not per candidate."""
    actual_san = before_board.san(move)
    after_board = before_board.copy()
    after_board.push(move)
    after_analysis = engine.analyse(after_board)
    after_engine = after_analysis.to_dict()
    after_facts = features.position_facts(after_board)
    value_after = _position_value_white(after_board, after_engine)

    # The full consequence: the move itself + the opponent's best reply, annotated
    # with captures and the NET material change measured from BEFORE the move — so a
    # capture that gets recaptured nets to an even trade, not a phantom "win".
    best_reply = after_analysis.lines[0].moves if after_analysis.lines else []
    # Annotate the WHOLE engine line (move + every reply ply): capping it shorter used
    # to drop the final recapture and report half an exchange as a huge material win.
    consequence = features.annotate_line(
        before_board, [move] + list(best_reply), max_plies=1 + len(best_reply)
    )
    # What the move changed positionally (move + best reply, so a trade's effect —
    # e.g. losing the bishop pair after the recapture — shows up).
    result_board = after_board.copy()
    if best_reply:
        result_board.push(best_reply[0])
    position_changes = features.positional_changes(before_board, result_board)

    # Judge the move: how far short of the engine's best was it, for the mover?
    best_list = before_analysis.to_dict()["best_moves"]
    engine_best_san = best_list[0]["move"] if best_list else None
    played_the_best = (
        engine_best_san is not None
        and _normalize_san(actual_san) == _normalize_san(engine_best_san)
    )
    mover = "white" if before_board.turn else "black"
    quality = _classify_move(mover, _white_value(best_list), value_after, played_the_best)
    actual_does = features.annotate_line(before_board, [move], max_plies=1)["moves"][0]
    return {
        "move": actual_san,
        "verdict": quality["verdict"],
        "centipawns_lost": quality["centipawns_lost"],
        "played_engine_best": played_the_best,
        "move_does": actual_does,
        # Named, verifiable tactics the move creates (forks, pins, pieces it hangs) —
        # the sharp "why", complementing the slower positional deltas below.
        "tactics": tactics.move_tactics(before_board, move),
        "consequence": consequence,
        "position_changes": position_changes,
        "engine_after": after_engine,
        "facts_after": after_facts,
        "value_white_after": value_after,
    }


def explain_move(move: str) -> dict:
    """Explain why a move PLAYED IN THE GAME was good or bad — your main tool.

    Works for ANY move, good or bad. Pass the move as the user refers to it, e.g.
    "Nxd4", "9.Nxd4", or "exd5"; you already have the whole game, so never ask
    the user to restate the position. In ONE call it returns:
      - `verdict`: best move / good move / inaccuracy / mistake / blunder,
      - `centipawns_lost`: how much worse than the engine's best (0 = best),
      - `engine_recommended`: the move the engine preferred, with its evaluation,
      - `recommended_plan`: the preferred move plus `plan` — each move of the
        engine's intended line annotated with exactly what it captures/attacks —
        and `achieves`; use this to teach the IDEA behind the right move precisely,
      - `actual_move_does`: whether the played move captures/gives check,
      - `tactics`: named, verifiable tactics the move creates — a fork, a pin, or a
        piece it leaves hanging (empty when the move makes no concrete tactic); this
        is the sharp reason a move wins or loses material,
      - `consequence`: the move + the opponent's best reply, annotated with
        captures and the NET material result (an equal trade nets to zero),
      - `position_changes`: positional things the move changed (e.g. "White gave
        up the bishop pair", "Black now has doubled pawns") — the grounded "why"
        for quiet moves; empty when nothing concrete changed,
      - `facts_before` / `facts_after`: material, hanging pieces, checks.
    Lead with the verdict. For a good/best move, affirm it and say what it
    achieves; for a weak move, give the better option and the punishment.
    """
    ctx = _ctx()
    session = ctx.session
    ply = _resolve_ply(session, move)
    if ply is None:
        result = {"ok": False, "error": f"'{move}' is not a move in this game — check the move list."}
        _log(f"explain_move({move!r}) -> {result['error']}")
        return result

    before = session.goto_move(ply - 1)  # mainline position just before the move
    before_board = session.current
    before_analysis = ctx.engine.analyse(before_board)  # Analysis object (keeps moves)
    before_engine = before_analysis.to_dict()
    ctx.session.analysis_cache[before_board.fen()] = before_engine
    before_facts = session.current_facts()
    best_list = before_engine["best_moves"]

    move_obj = session.mainline_moves[ply - 1]
    candidate = _evaluate_candidate(ctx.engine, before_board, before_analysis, move_obj)
    recommended_plan = _recommended_plan(before_board, before_analysis, best_list)

    _log(
        f"explain_move({ply}) actual={candidate['move']} verdict={candidate['verdict']} "
        f"(lost {candidate['centipawns_lost']}cp) best={best_list[0]['move'] if best_list else None}"
    )
    return {
        "ok": True,
        "move_explained": f"{candidate['move']} (ply {ply})",
        "verdict": candidate["verdict"],
        "centipawns_lost": candidate["centipawns_lost"],
        "actual_move": candidate["move"],
        "actual_move_does": candidate["move_does"],
        "tactics": candidate["tactics"],
        "played_engine_best": candidate["played_engine_best"],
        "engine_recommended": best_list[0] if best_list else None,
        "recommended_plan": recommended_plan,
        "consequence": candidate["consequence"],
        "position_changes": candidate["position_changes"],
        "position_before": before,
        "engine_before": before_engine,
        "facts_before": before_facts,
        "engine_after": candidate["engine_after"],
        "facts_after": candidate["facts_after"],
    }


def _ply_label(ply: int, san: str) -> str:
    """Render a ply as the user would refer to it: "18.Nxd4" / "18...Nxd4"."""
    full_move = (ply + 1) // 2
    return f"{full_move}.{san}" if ply % 2 == 1 else f"{full_move}...{san}"


def _swing_ply_window(
    around_move: Optional[int], total_plies: int, window: int = 2
) -> Optional[tuple[int, int]]:
    """The inclusive 1-based ply range to scan for `find_eval_swings`.

    `around_move is None` scans the whole game. Otherwise we centre on full-move
    M and scan ±`window` full moves (both colours): plies 2(M-window)-1 .. 2(M+window),
    clamped to the game. Returns None when there is nothing to scan — an empty game,
    or a move that lies entirely past the game's end (so the caller can say so)."""
    if total_plies < 1:
        return None
    if around_move is None:
        return (1, total_plies)
    if around_move < 1:
        return None
    start = 2 * (around_move - window) - 1
    end = 2 * (around_move + window)
    if start > total_plies:  # the requested move is past the end of the game
        return None
    return (max(1, start), min(total_plies, end))


_SCOPES = ("player", "opponent", "both")


def _side_of_ply(ply: int) -> str:
    """Which colour played the 1-based `ply` (odd = White, even = Black)."""
    return "white" if ply % 2 == 1 else "black"


def _resolve_scope(scope: Optional[str], player_color: Optional[str]) -> dict:
    """Turn the requested `scope` into the concrete one to scan, or an error.

    `scope is None` is the model omitting it: scan the user's own moves when we
    know their colour, else fall back to the whole game (so colourless sessions —
    e.g. a PGN piped in without `--color` — still work). An EXPLICIT player/opponent
    scope with no known colour is a clear error, never a silent both-sides scan."""
    if scope is None:
        return {"ok": True, "scope": "player" if player_color else "both"}
    if scope not in _SCOPES:
        return {"ok": False, "error": f"unknown scope {scope!r}: use 'player', 'opponent', or 'both'."}
    if scope in ("player", "opponent") and player_color is None:
        return {
            "ok": False,
            "error": (
                f"I don't know which side you played, so I can't scan only the "
                f"{'your' if scope == 'player' else 'opponent'} moves. Tell me whether "
                f"you were White or Black (or ask about the whole game)."
            ),
        }
    return {"ok": True, "scope": scope}


def _ply_in_scope(ply: int, scope: str, player_color: Optional[str]) -> bool:
    """Does the move at `ply` belong to the side this scan cares about?"""
    if scope == "both":
        return True
    side = _side_of_ply(ply)
    return side == player_color if scope == "player" else side != player_color


def _rank_swings(records: list[dict], max_results: int) -> list[dict]:
    """The worst `max_results` moves, biggest centipawn loss first.

    Records with no centipawn loss (None — only a position with no engine best
    move, which can't occur mid-mainline) are dropped: they can't be ranked.
    `max_results` is clamped to [1, 5] so the model can't ask for an unbounded dump."""
    capped = max(1, min(int(max_results), 5))
    rankable = [r for r in records if r["centipawns_lost"] is not None]
    rankable.sort(key=lambda r: r["centipawns_lost"], reverse=True)
    return rankable[:capped]


def find_eval_swings(
    max_results: int = 3, around_move: Optional[int] = None, scope: Optional[str] = None
) -> dict:
    """Find the moves where the game's evaluation dropped the most — the mistakes.

    Use this for VAGUE questions where the user has NOT named a move: "where did I
    go wrong?", "what was my biggest mistake?", "why did my eval drop?", "what was
    the turning point?". You no longer have to guess which move to inspect — this
    scans the real game and finds it for you. For "around move N", pass
    around_move=N to scope the scan to that part of the game (faster and focused);
    omit it to scan the whole game.

    `scope` picks WHOSE moves to scan — this matters because the user played only
    ONE side, so "my biggest mistake" must not return the opponent's:
      - "player" (the DEFAULT): only the USER's own moves — use for "where did I go
        wrong?", "what was my biggest mistake?", "did I blunder?".
      - "opponent": only the other side's moves — use for "what did my opponent
        miss?", "where did they go wrong?".
      - "both": every move — use for neutral questions like "what was the turning
        point of the game?".
    If the user's colour is unknown, an explicit "player"/"opponent" scope returns
    an error (ok=false) — relay it and ask which side they played; do NOT pretend.

    Returns `swings`: the worst `max_results` (1-3) moves, BIGGEST drop first. Each
    entry has the SAME fields as explain_move — `verdict`, `centipawns_lost`,
    `engine_recommended`, `recommended_plan`, `actual_move_does`, `tactics`,
    `consequence`, `position_changes`, `facts_before`/`facts_after`, plus `move_label` (e.g.
    "18.Nxd4"), `ply`, `side` (white/black), and `eval_before_white_cp`/
    `eval_after_white_cp` (White's POV) — so present each swing exactly as you would
    a single explained move, and lead with the biggest one. `scanned.scope` echoes
    whose moves were examined. `note` flags a cleanly played game (only small drops)
    or a move number past the game's end. (When the engine's best is a forced mate,
    that move's `centipawns_lost` is huge by design — it correctly ranks as the
    biggest swing; trust the `verdict`.)
    """
    ctx = _ctx()
    session = ctx.session

    resolved = _resolve_scope(scope, session.player_color)
    if not resolved["ok"]:
        _log(f"find_eval_swings(scope={scope!r}) -> {resolved['error']}")
        return resolved
    scope = resolved["scope"]

    total_plies = len(session.mainline_moves)
    total_full_moves = (total_plies + 1) // 2

    window = _swing_ply_window(around_move, total_plies)
    if window is None:
        note = (
            f"move {around_move} is beyond this game (it has {total_full_moves} moves)."
            if around_move is not None
            else "this game has no moves to scan."
        )
        _log(f"find_eval_swings(around_move={around_move}) -> {note}")
        return {"ok": True,
                "scanned": {"from_ply": None, "to_ply": None, "moves_examined": 0, "scope": scope},
                "swings": [], "note": note}

    from_ply, to_ply = window
    records: list[dict] = []
    board = chess.Board(session.start_fen)  # our own board — the session is never touched
    for idx, move_obj in enumerate(session.mainline_moves):
        ply = idx + 1
        # Only the side this scan cares about gets the (costly) engine analysis;
        # every move is still pushed below so later positions stay correct.
        if from_ply <= ply <= to_ply and _ply_in_scope(ply, scope, session.player_color):
            before_analysis = ctx.engine.analyse(board)
            before_engine = before_analysis.to_dict()
            session.analysis_cache[board.fen()] = before_engine  # warms a later explain_move
            best_list = before_engine["best_moves"]
            candidate = _evaluate_candidate(ctx.engine, board, before_analysis, move_obj)
            records.append({
                "ply": ply,
                "move_label": _ply_label(ply, session.mainline_san[idx]),
                "side": "white" if ply % 2 == 1 else "black",
                "verdict": candidate["verdict"],
                "centipawns_lost": candidate["centipawns_lost"],
                "actual_move": candidate["move"],
                "actual_move_does": candidate["move_does"],
                "tactics": candidate["tactics"],
                "played_engine_best": candidate["played_engine_best"],
                "engine_recommended": best_list[0] if best_list else None,
                "recommended_plan": _recommended_plan(board, before_analysis, best_list),
                "consequence": candidate["consequence"],
                "position_changes": candidate["position_changes"],
                "facts_before": features.position_facts(board),
                "facts_after": candidate["facts_after"],
                "eval_before_white_cp": _white_value(best_list),
                "eval_after_white_cp": candidate["value_white_after"],
            })
        board.push(move_obj)

    swings = _rank_swings(records, max_results)
    if not swings:
        note = "no rankable moves in this range."
    elif swings[0]["centipawns_lost"] < 50:
        note = (
            f"the biggest drop was only {swings[0]['centipawns_lost']}cp "
            f"({swings[0]['move_label']}) — a cleanly played game, no real mistakes."
        )
    else:
        top = swings[0]
        note = f"biggest drop: {top['move_label']} lost {top['centipawns_lost']}cp ({top['verdict']})."

    _log(
        f"find_eval_swings(scope={scope}, plies {from_ply}-{to_ply}, examined {len(records)}) -> "
        f"top={swings[0]['move_label'] if swings else None} "
        f"lost={swings[0]['centipawns_lost'] if swings else None}cp"
    )
    return {
        "ok": True,
        "scanned": {"from_ply": from_ply, "to_ply": to_ply,
                    "moves_examined": len(records), "scope": scope},
        "swings": swings,
        "note": note,
    }


def _to_mover_value(value_white: Optional[int], mover: str) -> Optional[int]:
    """Re-sign a White-POV value to the mover's POV (higher = better for them)."""
    if value_white is None:
        return None
    return value_white if mover == "white" else -value_white


def _compare_verdict(candidate_a: dict, candidate_b: dict, mover: str) -> dict:
    """Which candidate the engine rates higher for the mover, and by how much.

    Differences within 15cp are called "about equal" (engine noise). When a line
    forces mate, a centipawn gap is meaningless, so we report the winner with a
    note instead of a number.
    """
    va = _to_mover_value(candidate_a["value_white_after"], mover)
    vb = _to_mover_value(candidate_b["value_white_after"], mover)
    if va is None or vb is None:
        return {
            "better_move": "unclear",
            "centipawn_gap": None,
            "note": "a line had no centipawn score (e.g. it forces mate) — judge by the verdicts.",
        }
    better = candidate_a["move"] if va > vb else candidate_b["move"] if vb > va else "about equal"
    if abs(va) >= _MATE_VALUE or abs(vb) >= _MATE_VALUE:
        return {
            "better_move": better,
            "centipawn_gap": None,
            "note": "the difference is a forced mate, not a centipawn count.",
        }
    gap = abs(va - vb)
    return {"better_move": "about equal" if gap <= 15 else better, "centipawn_gap": gap}


def compare_moves(move_a: str, move_b: str) -> dict:
    """Compare TWO candidate moves from the CURRENT position and say which is better.

    Use this when the user asks which of two moves is better, or WHY one move is
    better than another (e.g. "is Nf3 or Nc3 better here?", "why is Bd3 better than
    Be2?"). First call goto_move(ply) so the board sits at the position the choice
    is from — it must be that player's turn — then call compare_moves with the two
    moves in algebraic notation ('Nf3', 'exd5', 'O-O', or UCI 'e2e4').

    Both moves are scored from the SAME position, so the comparison is fair. The
    result has, for EACH move (`move_a`, `move_b`): a `verdict` (best move / good
    move / inaccuracy / mistake / blunder), `centipawns_lost` vs the engine's best,
    `move_does` (what it captures / whether it checks), `tactics` (any fork / pin /
    hung piece the move creates), `consequence` (the move + the opponent's best reply
    with the NET material result), and `position_changes`
    (what it changed positionally). It also returns `comparison.better_move` and
    `comparison.centipawn_gap` (how much better, from the mover's point of view) and
    `engine_best` — the engine's own top pick here with its plan, in case NEITHER
    move was best. Lead with which move is better and the gap, then explain WHY from
    each move's plan / position_changes / consequence, the same way you explain a
    single move. If a move is illegal here, relay the reason and do NOT invent a line.
    """
    ctx = _ctx()
    session = ctx.session
    board = session.current
    label = f"compare_moves({move_a!r}, {move_b!r})"
    if board.is_game_over(claim_draw=True):
        result = {"ok": False, "error": "this position is already over — there are no moves to compare."}
        _log(f"{label} -> {result['error']}")
        return result

    mover = "white" if board.turn else "black"
    parsed: dict[str, chess.Move] = {}
    for name, text in (("move_a", move_a), ("move_b", move_b)):
        obj = session.parse_move(text)
        if obj is None:
            result = {
                "ok": False,
                "error": f"illegal: '{text}' is not a legal move in this position (side to move: {mover})",
            }
            _log(f"{label} -> {result['error']}")
            return result
        parsed[name] = obj
    if parsed["move_a"] == parsed["move_b"]:
        result = {"ok": False, "error": "those are the same move — give two different moves to compare."}
        _log(f"{label} -> {result['error']}")
        return result

    before_analysis = ctx.engine.analyse(board)
    before_engine = before_analysis.to_dict()
    ctx.session.analysis_cache[board.fen()] = before_engine
    best_list = before_engine["best_moves"]

    candidate_a = _evaluate_candidate(ctx.engine, board, before_analysis, parsed["move_a"])
    candidate_b = _evaluate_candidate(ctx.engine, board, before_analysis, parsed["move_b"])
    comparison = _compare_verdict(candidate_a, candidate_b, mover)

    _log(
        f"compare_moves({candidate_a['move']} vs {candidate_b['move']}) -> "
        f"better={comparison['better_move']} gap={comparison['centipawn_gap']}"
    )
    return {
        "ok": True,
        "position": session.orientation(),
        "facts_before": session.current_facts(),
        "engine_best": _recommended_plan(board, before_analysis, best_list),
        "move_a": candidate_a,
        "move_b": candidate_b,
        "comparison": comparison,
    }


def play_move(move: str) -> dict:
    """Play a HYPOTHETICAL move from the current position (a "what if").

    `move` is in algebraic notation: e.g. 'Nf3', 'exd5', 'O-O', or UCI 'e2e4'.
    This branches off a copy, so the user's real game is never changed. If the
    move is illegal the result has ok=false and an error explaining why — relay
    that reason to the user and do NOT invent a line. Call analyze() afterwards
    to evaluate the resulting position.
    """
    result = _ctx().session.play_move(move)
    _log(f"play_move({move!r}) -> {_pos_summary(result)}")
    return result


def go_back() -> dict:
    """Undo the last hypothetical move and return to the previous position.

    Use this when you are done exploring a what-if branch.
    """
    result = _ctx().session.go_back()
    _log(f"go_back() -> {_pos_summary(result)}")
    return result


# ---- probes: general questions the model asks to test its own hypotheses ------
#
# The verdict tools above hand the model pre-named reasons (fork, pin, bishop pair…),
# which caps its explanations at the motifs we wrote detectors for. These probes are
# the opposite: small, general, always-true questions about the board, so the model
# can investigate like a coach (form a hypothesis, check it) and name ideas we never
# hand-coded — without ever stating a fact a tool didn't return. None of them moves
# the board: they read `session.current` and leave it where it was.

_MAX_LINE_PLIES = 10  # long enough for a real combination, short enough to stay precise
_SERIOUS_THREAT_CP = 100  # a threat worth about a pawn or more is worth teaching


def _parse_line(board: chess.Board, text: str) -> tuple[list[chess.Move], Optional[str]]:
    """Parse a space-separated line ("12.Nxe5 Qxe5 13.Rxd8+") into moves legal in
    sequence from `board`. Returns (moves, None) or ([], error) — the first illegal
    move is named with its position in the line so the model can't build on it."""
    tokens = [t for t in re.split(r"[\s,]+", text.strip()) if t]
    tokens = [t for t in tokens if not re.fullmatch(r"\d+\.+", t)]  # bare "12." / "12..."
    if not tokens:
        return [], "no moves given — pass a line like 'Nxe5 Qxe5 Rxd8+'"
    if len(tokens) > _MAX_LINE_PLIES:
        return [], f"that line has {len(tokens)} half-moves — give at most {_MAX_LINE_PLIES}"
    work = board.copy()
    moves: list[chess.Move] = []
    for index, token in enumerate(tokens, start=1):
        text_move = _normalize_san(token)
        move = None
        for parse in (work.parse_san, work.parse_uci):
            try:
                candidate = parse(text_move)
            except ValueError:  # python-chess's illegal/invalid/ambiguous errors
                continue
            if candidate in work.legal_moves:
                move = candidate
                break
        if move is None:
            side = "white" if work.turn else "black"
            return [], (
                f"illegal: move {index} of the line, '{token}', is not legal there "
                f"(side to move: {side}) — nothing after it was played"
            )
        moves.append(move)
        work.push(move)
    return moves, None


def _walk_line(board: chess.Board, moves: list[chess.Move]) -> dict:
    """`annotate_line` plus what is left hanging and whether it is mate after EACH
    move — so the model can point at the exact ply where material falls instead of
    only the net result."""
    walked = features.annotate_line(board, moves, max_plies=len(moves))
    work = board.copy()
    for step, move in zip(walked["moves"], moves):
        work.push(move)
        step["hanging_after"] = features.hanging_pieces(work)
        step["checkmate"] = work.is_checkmate()
    return walked


def _cached_analysis(board: chess.Board) -> dict:
    """Engine verdict on `board`, reusing the session cache (analysis is the slow part)."""
    ctx = _ctx()
    fen = board.fen()
    analysis = ctx.session.analysis_cache.get(fen)
    if analysis is None:
        analysis = ctx.engine.analyse(board).to_dict()
        ctx.session.analysis_cache[fen] = analysis
    return analysis


def square_info(square: str) -> dict:
    """Probe ONE square on the CURRENT board — your magnifying glass for testing an
    idea before you say it. `square` is like "e5".

    Returns the piece on it (if any) and: `attacked_by` and `defended_by` (every piece
    that directly hits the square, each marked `pinned_to_king` — a pinned piece can't
    really capture or recapture), `attacks` (enemy pieces this piece hits), `defends`
    (its own pieces this piece protects), `pinned_to_king`, and
    `attacked_by_cheaper_piece` (it loses material even if defended). For an empty
    square you get `attacked_by_white` / `attacked_by_black`.

    Use it to CHECK hypotheses: an overloaded defender (one piece in `defends` for two
    things that are both attacked), a defender that is pinned, a piece attacked by a
    cheaper piece, a square the opponent controls. Only claim what it returns.
    """
    ctx = _ctx()
    try:
        parsed = chess.parse_square(square.strip().lower())
    except ValueError:
        result = {"ok": False, "error": f"'{square}' is not a square — use a name like 'e5'."}
        _log(f"square_info({square!r}) -> {result['error']}")
        return result
    report = features.square_report(ctx.session.current, parsed)
    occupant = report["piece"]
    label = f"{occupant['color']} {occupant['type']}" if occupant else "empty"
    _log(f"square_info({report['square']}) -> {label}")
    return {"ok": True, "position": ctx.session.orientation(), **report}


def try_line(moves: str) -> dict:
    """Walk a sequence of moves from the CURRENT position WITHOUT moving the board,
    and see exactly what happens at every step. `moves` is a space-separated line in
    algebraic notation, e.g. "Nxe5 Qxe5 Rxd8+" (move numbers are fine; at most 10).

    Returns `line.moves`: for EACH move — who played it, what it `captures`, which
    pieces it `attacks`, whether it `gives_check`, `hanging_after` (pieces left
    attacked and undefended after that move), and `checkmate` — plus the net material
    result, and `engine_after_line` (the evaluation where the line ends). If any move
    is illegal you get ok=false naming it; do NOT describe the line past that point.

    Use it to walk the engine's line (e.g. the played move followed by the moves in
    `consequence`) and find the precise moment material falls or a threat appears, or
    to test "what if they had played X here?" in more than one move.
    """
    ctx = _ctx()
    board = ctx.session.current
    parsed, error = _parse_line(board, moves)
    if error:
        result = {"ok": False, "error": error}
        _log(f"try_line({moves!r}) -> {error}")
        return result
    walked = _walk_line(board, parsed)
    end = board.copy()
    for move in parsed:
        end.push(move)
    analysis = _cached_analysis(end)
    _log(f"try_line({moves!r}) -> {walked['summary']} | {_eval_summary(analysis)}")
    return {
        "ok": True,
        "position": ctx.session.orientation(),
        "line": walked,
        "engine_after_line": analysis,
        "facts_after_line": features.position_facts(end),
    }


def threats() -> dict:
    """What is the opponent THREATENING in the CURRENT position? Answers "what would
    they play if it were their move right now?" by letting the side to move pass.

    Returns `threat.move` (the opponent's best move if ignored), `threat.line` (that
    line annotated with captures / attacks / checks), `threat.tactics` (a fork, pin or
    hung piece it creates), `threat.eval_if_ignored`, `threat.threatens_mate` (with
    `mate_in`), and `threat.size_centipawns` — how much the opponent gains if the
    threat is ignored (empty for a mate threat) — with `threat.serious` (true for a
    mate threat or a gain of about a pawn or more). `current_engine` is the real
    evaluation.

    Use it for "what did I miss?", "why did I have to defend there?", or when a quiet
    move lost because it ignored something. If `serious` is false, say there was no
    real threat rather than inventing one. Not available while in check (the check
    IS the threat) — call analyze() then.
    """
    ctx = _ctx()
    board = ctx.session.current
    side = "white" if board.turn else "black"
    opponent = "black" if board.turn else "white"
    if board.is_game_over(claim_draw=True):
        result = {"ok": False, "error": "the game is over in this position — there is no threat to find."}
        _log(f"threats() -> {result['error']}")
        return result
    if board.is_check():
        result = {
            "ok": False,
            "error": f"{side} is in check — the check itself is the threat; call analyze() instead.",
        }
        _log(f"threats() -> {result['error']}")
        return result

    current = _cached_analysis(board)
    # A "null move" hands the turn to the opponent without changing the board — the
    # standard engine trick for asking what they want to do. Legal here because the
    # side to move is not in check, so the resulting position is valid. We rebuild it
    # from the FEN so the engine gets a clean position with no null move in its history.
    passed_with_history = board.copy(stack=False)
    passed_with_history.push(chess.Move.null())
    passed = chess.Board(passed_with_history.fen())
    passed_analysis = ctx.engine.analyse(passed)
    if not passed_analysis.lines or not passed_analysis.lines[0].moves:
        result = {"ok": False, "error": f"the engine found no move for {opponent} here — no threat to report."}
        _log(f"threats() -> {result['error']}")
        return result
    top = passed_analysis.lines[0]
    top_dict = top.to_dict()
    threatens_mate = top.mate_in is not None and (top.mate_in > 0) == (opponent == "white")
    value_now = _white_value(current["best_moves"])
    value_ignored = _white_value([top_dict])
    size = None
    # A mate threat has no honest centipawn size (mate and centipawns share one scale,
    # so the difference would read as ~100000) — report `mate_in` and leave size empty.
    mate_involved = any(v is not None and abs(v) >= _MATE_VALUE for v in (value_now, value_ignored))
    if not mate_involved and value_now is not None and value_ignored is not None:
        gain = (value_ignored - value_now) if opponent == "white" else (value_now - value_ignored)
        size = max(int(gain), 0)
    serious = threatens_mate or (size is not None and size >= _SERIOUS_THREAT_CP)
    _log(f"threats() -> {opponent} threatens {top.move} ({top_dict['eval']}, size={size}, serious={serious})")
    return {
        "ok": True,
        "position": ctx.session.orientation(),
        "threatening_side": opponent,
        "threat": {
            "move": top.move,
            "line": features.annotate_line(passed, top.moves, max_plies=6),
            "tactics": tactics.move_tactics(passed, top.moves[0]),
            "eval_if_ignored": top_dict["eval"],
            "mate_in": top.mate_in,
            "threatens_mate": threatens_mate,
            "size_centipawns": size,
            "serious": serious,
        },
        "current_engine": current,
    }


# The exact list handed to google-genai as `config.tools`.
ALL_TOOLS = [
    explain_move, compare_moves, find_eval_swings, explain_position,
    goto_move, analyze, play_move, go_back,
    square_info, try_line, threats,
]
