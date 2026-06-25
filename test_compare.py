"""Tests for compare_moves and its grounded-comparison helpers."""

import io

import chess
import chess.pgn
import pytest

import tools
from engine import Engine, EngineError, find_stockfish
from session import SessionState

try:
    find_stockfish()
    HAVE_STOCKFISH = True
except EngineError:
    HAVE_STOCKFISH = False

needs_engine = pytest.mark.skipif(not HAVE_STOCKFISH, reason="stockfish not installed")


# ---- pure-logic helpers (no engine) ---------------------------------------


def test_position_value_white_reads_checkmate_as_mate_for_the_loser():
    board = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4#"]:  # fool's mate — White is mated, White to move
        board.push_san(san)
    assert board.is_checkmate() and board.turn == chess.WHITE
    value = tools._position_value_white(board, {"best_moves": []})
    assert value == -tools._MATE_VALUE  # the side to move (White) is the one mated


def test_position_value_white_reads_stalemate_as_zero():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # Black to move, no legal move
    assert board.is_stalemate()
    assert tools._position_value_white(board, {"best_moves": []}) == 0


def test_compare_verdict_prefers_higher_mover_value():
    a = {"move": "Nf3", "value_white_after": 80}
    b = {"move": "Nc3", "value_white_after": 20}
    assert tools._compare_verdict(a, b, "white")["better_move"] == "Nf3"
    # From Black's POV the signs flip: the lower White value is better for Black.
    assert tools._compare_verdict(a, b, "black")["better_move"] == "Nc3"


def test_compare_verdict_calls_near_equal_a_wash():
    a = {"move": "Nf3", "value_white_after": 30}
    b = {"move": "Nc3", "value_white_after": 25}
    out = tools._compare_verdict(a, b, "white")
    assert out["better_move"] == "about equal"
    assert out["centipawn_gap"] == 5


def test_compare_verdict_flags_forced_mate_gap():
    a = {"move": "Qh4", "value_white_after": tools._MATE_VALUE}
    b = {"move": "d5", "value_white_after": 40}
    out = tools._compare_verdict(a, b, "white")
    assert out["better_move"] == "Qh4"
    assert out["centipawn_gap"] is None
    assert "mate" in out["note"]


def test_compare_verdict_unclear_when_value_missing():
    a = {"move": "Nf3", "value_white_after": None}
    b = {"move": "Nc3", "value_white_after": 40}
    assert tools._compare_verdict(a, b, "white")["better_move"] == "unclear"


# ---- end-to-end through the engine ----------------------------------------


def _session(pgn: str) -> SessionState:
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    return SessionState.from_game(game)


@needs_engine
def test_compare_moves_picks_the_mate():
    session = _session("1. f3 e5 2. g4 d5 0-1")
    session.goto_move(3)  # after 1.f3 e5 2.g4 — Black to move, Qh4# is forced mate
    with Engine() as engine:
        tools.bind(session, engine)
        result = tools.compare_moves("Qh4", "d5")
    assert result["ok"] is True
    assert result["comparison"]["better_move"].rstrip("+#") == "Qh4"
    assert result["move_a"]["verdict"] == "best move"
    assert result["move_b"]["centipawns_lost"] > 0


@needs_engine
def test_compare_moves_rejects_illegal_move():
    session = _session("1. e4 e5 0-1")
    session.goto_move(0)  # starting position
    with Engine() as engine:
        tools.bind(session, engine)
        result = tools.compare_moves("e4", "Qh4")  # Qh4 is illegal from the start
    assert result["ok"] is False
    assert result["error"].startswith("illegal:")


@needs_engine
def test_compare_moves_rejects_identical_moves():
    session = _session("1. e4 e5 0-1")
    session.goto_move(0)
    with Engine() as engine:
        tools.bind(session, engine)
        result = tools.compare_moves("e4", "e2e4")  # the same move in two notations
    assert result["ok"] is False
    assert "same move" in result["error"]


@needs_engine
def test_compare_moves_leaves_current_position_unchanged():
    session = _session("1. e4 e5 2. Nf3 Nc6 1-0")
    session.goto_move(2)  # after 1.e4 e5, White to move
    fen_before = session.current.fen()
    with Engine() as engine:
        tools.bind(session, engine)
        tools.compare_moves("Nf3", "Nc3")
    assert session.current.fen() == fen_before  # comparison must not move the board


# ---- find_eval_swings: pure helpers (no engine) ---------------------------


def test_swing_ply_window_none_scans_whole_game():
    assert tools._swing_ply_window(None, total_plies=10) == (1, 10)


def test_swing_ply_window_empty_game_is_none():
    assert tools._swing_ply_window(None, total_plies=0) is None


def test_swing_ply_window_centres_and_clamps_on_a_move():
    # ±2 full moves around move 6 = plies 2*4-1 .. 2*8 = 7..16, clamped to the game.
    assert tools._swing_ply_window(6, total_plies=40, window=2) == (7, 16)
    # Near the start the lower bound clamps to ply 1.
    assert tools._swing_ply_window(1, total_plies=40, window=2) == (1, 6)
    # Near the end the upper bound clamps to the last ply.
    assert tools._swing_ply_window(20, total_plies=40, window=2) == (35, 40)


def test_swing_ply_window_beyond_game_is_none():
    assert tools._swing_ply_window(40, total_plies=10) is None
    assert tools._swing_ply_window(0, total_plies=10) is None


def test_rank_swings_sorts_descending_and_caps():
    records = [
        {"centipawns_lost": 30},
        {"centipawns_lost": 300},
        {"centipawns_lost": 120},
    ]
    ranked = tools._rank_swings(records, max_results=2)
    assert [r["centipawns_lost"] for r in ranked] == [300, 120]


def test_rank_swings_drops_unrankable_and_clamps_max_results():
    records = [{"centipawns_lost": None}, {"centipawns_lost": 50}, {"centipawns_lost": 10}]
    # max_results below 1 is clamped up to 1; the None record is dropped.
    ranked = tools._rank_swings(records, max_results=0)
    assert ranked == [{"centipawns_lost": 50}]
    # max_results above 5 is clamped down to 5.
    assert len(tools._rank_swings([{"centipawns_lost": i} for i in range(9)], 99)) == 5


# ---- find_eval_swings: scope resolution (no engine) -----------------------


def test_resolve_scope_default_is_player_when_color_known():
    assert tools._resolve_scope(None, "white") == {"ok": True, "scope": "player"}


def test_resolve_scope_default_is_both_when_color_unknown():
    # An OMITTED scope on a colourless session falls back to scanning everything,
    # so games loaded without a known colour still work.
    assert tools._resolve_scope(None, None) == {"ok": True, "scope": "both"}


def test_resolve_scope_rejects_unknown_scope():
    out = tools._resolve_scope("mine", "white")
    assert out["ok"] is False and "unknown scope" in out["error"]


def test_resolve_scope_errors_on_explicit_player_scope_without_color():
    for scope in ("player", "opponent"):
        out = tools._resolve_scope(scope, None)
        assert out["ok"] is False
        assert "which side you played" in out["error"]


def test_ply_in_scope_filters_by_side():
    # White plays odd plies, Black even.
    assert tools._ply_in_scope(1, "player", "white") is True
    assert tools._ply_in_scope(2, "player", "white") is False
    assert tools._ply_in_scope(2, "opponent", "white") is True
    assert tools._ply_in_scope(1, "opponent", "white") is False
    assert tools._ply_in_scope(1, "both", None) is True
    assert tools._ply_in_scope(2, "both", None) is True


def test_find_eval_swings_invalid_scope_returns_clear_error():
    session = _session("1. e4 e5 2. Nf3 Nc6 1-0")
    tools.bind(session, object())  # the error path never touches the engine
    result = tools.find_eval_swings(scope="mine")
    assert result["ok"] is False
    assert "unknown scope" in result["error"]


def test_find_eval_swings_player_scope_without_color_errors():
    session = _session("1. e4 e5 2. Nf3 Nc6 1-0")  # no colour set
    tools.bind(session, object())
    result = tools.find_eval_swings(scope="player")
    assert result["ok"] is False
    assert "which side you played" in result["error"]


# ---- find_eval_swings: end-to-end through the engine ----------------------


@needs_engine
def test_find_eval_swings_finds_the_blunder_sorted_worst_first():
    session = _session("1. f3 e5 2. g4 Qh4# 0-1")  # White's g4 walks into mate
    with Engine() as engine:
        tools.bind(session, engine)
        result = tools.find_eval_swings()
    assert result["ok"] is True
    swings = result["swings"]
    assert swings, "a game ending in fool's mate must surface at least one swing"
    losses = [s["centipawns_lost"] for s in swings]
    assert losses == sorted(losses, reverse=True)  # biggest drop first
    assert swings[0]["verdict"] in {"mistake", "blunder"}
    # Each swing carries the explain_move-shaped payload the prompt relies on.
    for key in (
        "engine_recommended", "recommended_plan", "consequence",
        "position_changes", "facts_before", "facts_after", "move_label",
    ):
        assert key in swings[0]


@needs_engine
def test_find_eval_swings_respects_max_results():
    session = _session("1. f3 e5 2. g4 Qh4# 0-1")
    with Engine() as engine:
        tools.bind(session, engine)
        result = tools.find_eval_swings(max_results=1)
    assert len(result["swings"]) <= 1


@needs_engine
def test_find_eval_swings_around_move_scopes_the_scan():
    session = _session("1. f3 e5 2. g4 Qh4# 0-1")
    with Engine() as engine:
        tools.bind(session, engine)
        result = tools.find_eval_swings(around_move=2)
    scanned = result["scanned"]
    assert scanned["from_ply"] is not None and scanned["to_ply"] is not None
    for swing in result["swings"]:
        assert scanned["from_ply"] <= swing["ply"] <= scanned["to_ply"]


@needs_engine
def test_find_eval_swings_beyond_game_reports_it():
    session = _session("1. f3 e5 2. g4 Qh4# 0-1")  # 2 full moves
    with Engine() as engine:
        tools.bind(session, engine)
        result = tools.find_eval_swings(around_move=40)
    assert result["ok"] is True
    assert result["swings"] == []
    assert "beyond this game" in result["note"]


@needs_engine
def test_find_eval_swings_leaves_session_untouched():
    session = _session("1. f3 e5 2. g4 Qh4# 0-1")
    session.goto_move(2)
    fen_before = session.current.fen()
    san_before = list(session.mainline_san)
    with Engine() as engine:
        tools.bind(session, engine)
        tools.find_eval_swings()
    assert session.current.fen() == fen_before  # the user's cursor must not move
    assert session.mainline_san == san_before  # the mainline is read-only


# ---- find_eval_swings: player-aware scope (end-to-end through the engine) --


@needs_engine
def test_find_eval_swings_player_scope_scans_only_the_users_moves():
    session = _session("1. f3 e5 2. g4 Qh4# 0-1")
    session.set_player_color("black")
    with Engine() as engine:
        tools.bind(session, engine)
        result = tools.find_eval_swings(scope="player")
    assert result["scanned"]["scope"] == "player"
    assert result["scanned"]["moves_examined"] == 2  # only Black's two moves examined
    assert all(s["side"] == "black" for s in result["swings"])


@needs_engine
def test_find_eval_swings_opponent_scope_scans_only_opponent_moves():
    session = _session("1. f3 e5 2. g4 Qh4# 0-1")
    session.set_player_color("black")  # the opponent is White here
    with Engine() as engine:
        tools.bind(session, engine)
        result = tools.find_eval_swings(scope="opponent")
    assert result["scanned"]["scope"] == "opponent"
    assert result["scanned"]["moves_examined"] == 2  # only White's two moves
    assert all(s["side"] == "white" for s in result["swings"])


@needs_engine
def test_find_eval_swings_both_scope_scans_every_move():
    session = _session("1. f3 e5 2. g4 Qh4# 0-1")
    session.set_player_color("black")
    with Engine() as engine:
        tools.bind(session, engine)
        result = tools.find_eval_swings(scope="both")
    assert result["scanned"]["scope"] == "both"
    assert result["scanned"]["moves_examined"] == 4  # all four half-moves


@needs_engine
def test_my_biggest_mistake_excludes_a_bigger_opponent_blunder():
    # White's 2.g4 walks into mate — the game's biggest blunder belongs to the
    # OPPONENT when the user played Black. "My biggest mistake" (scope="player")
    # must still return only the user's (Black's) moves, never that g4.
    session = _session("1. f3 e5 2. g4 Qh4# 0-1")
    session.set_player_color("black")
    with Engine() as engine:
        tools.bind(session, engine)
        result = tools.find_eval_swings(scope="player")
    assert all(s["side"] == "black" for s in result["swings"])
    assert all("g4" not in s["move_label"] for s in result["swings"])


@needs_engine
def test_find_eval_swings_player_scope_leaves_session_untouched():
    session = _session("1. f3 e5 2. g4 Qh4# 0-1")
    session.set_player_color("white")
    session.goto_move(2)
    fen_before = session.current.fen()
    san_before = list(session.mainline_san)
    with Engine() as engine:
        tools.bind(session, engine)
        tools.find_eval_swings(scope="player")
    assert session.current.fen() == fen_before
    assert session.mainline_san == san_before
