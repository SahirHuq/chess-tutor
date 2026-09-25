"""Tests for the investigation probes: square_info, try_line and threats."""

import io

import chess
import chess.pgn
import pytest

import features
import tools
from engine import Engine, EngineError, find_stockfish
from session import SessionState

try:
    find_stockfish()
    HAVE_STOCKFISH = True
except EngineError:
    HAVE_STOCKFISH = False

needs_engine = pytest.mark.skipif(not HAVE_STOCKFISH, reason="stockfish not installed")

# 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? — after 3.Qh5 White threatens Qxf7#; after 3...Nf6
# Black has ignored it and Qxf7# is on the board.
_SCHOLARS = "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 *"


def _session(pgn: str) -> SessionState:
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    return SessionState.from_game(game)


def _squares(entries: list[dict]) -> list[str]:
    return [e["square"] for e in entries]


# ---- square_report (pure, no engine) ----------------------------------------


def test_square_report_flags_a_defended_piece_attacked_by_a_cheaper_piece():
    # White knight on e5: hit by the d6 pawn, defended by the d4 pawn. The strict
    # hanging check ignores it (it IS defended) — this probe must not.
    board = chess.Board("4k3/8/3p4/4N3/3P4/8/8/4K3 w - - 0 1")
    report = features.square_report(board, chess.E5)
    assert report["piece"] == {"type": "knight", "color": "white"}
    assert _squares(report["attacked_by"]) == ["d6"]
    assert _squares(report["defended_by"]) == ["d4"]
    assert report["attacked_by_cheaper_piece"] is True
    assert features.hanging_pieces(board) == []  # the old detector can't see it


def test_square_report_marks_a_pinned_defender():
    # The c6 knight defends e5 but is pinned to the e8 king by the b5 bishop.
    board = chess.Board("4k3/8/2n5/1B2p3/8/8/8/4K3 w - - 0 1")
    report = features.square_report(board, chess.E5)
    (defender,) = report["defended_by"]
    assert defender["square"] == "c6" and defender["pinned_to_king"] is True


def test_square_report_lists_what_a_piece_defends_and_attacks():
    # Black queen d8 protects the a8 rook (clear rank) and hits the d1 rook; its own
    # king on e8 is not listed as a "defended" piece.
    board = chess.Board("r2qk3/8/8/8/8/8/8/3RK3 b - - 0 1")
    report = features.square_report(board, chess.D8)
    assert _squares(report["defends"]) == ["a8"]
    assert _squares(report["attacks"]) == ["d1"]


def test_square_report_on_an_empty_square_splits_attackers_by_colour():
    report = features.square_report(chess.Board(), chess.E3)
    assert report["piece"] is None
    assert sorted(_squares(report["attacked_by_white"])) == ["d2", "f2"]
    assert report["attacked_by_black"] == []


# ---- line parsing / walking (pure, no engine) --------------------------------


def test_parse_line_accepts_move_numbers_and_leaves_the_board_alone():
    board = chess.Board()
    moves, error = tools._parse_line(board, "1. e4 e5 2.Nf3 Nc6")
    assert error is None
    assert [m.uci() for m in moves] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert board.fen() == chess.STARTING_FEN


def test_parse_line_names_the_first_illegal_move():
    moves, error = tools._parse_line(chess.Board(), "e4 e4")
    assert moves == []
    assert "move 2" in error and "'e4'" in error


@pytest.mark.parametrize("text", ["", "   ", "1. 2."])
def test_parse_line_rejects_an_empty_line(text):
    moves, error = tools._parse_line(chess.Board(), text)
    assert moves == [] and "no moves given" in error


def test_parse_line_caps_the_length():
    line = "Nf3 Nf6 Ng1 Ng8 " * 3  # 12 legal half-moves
    moves, error = tools._parse_line(chess.Board(), line)
    assert moves == [] and "at most" in error


def test_walk_line_reports_hanging_pieces_and_mate_per_move():
    board = chess.Board()
    moves, _ = tools._parse_line(board, "e4 e5 Qh5 Nc6 Bc4 Nf6 Qxf7#")
    walked = tools._walk_line(board, moves)
    steps = walked["moves"]
    # After 2.Qh5 the e5 pawn is attacked and has no defender.
    assert "e5" in [h["square"] for h in steps[2]["hanging_after"]]
    assert steps[-1]["move"] == "Qxf7#" and steps[-1]["checkmate"] is True
    assert steps[-1]["captures"] == "pawn"


# ---- tool error paths (no engine touched) ------------------------------------


def test_square_info_rejects_a_bad_square():
    tools.bind(_session(_SCHOLARS), object())
    result = tools.square_info("z9")
    assert result["ok"] is False and "not a square" in result["error"]


def test_try_line_reports_illegal_moves_without_touching_the_engine():
    session = _session(_SCHOLARS)
    tools.bind(session, object())  # an illegal line must fail before any analysis
    result = tools.try_line("e5")  # White to move at the start; e5 is Black's move
    assert result["ok"] is False and "move 1" in result["error"]


# ---- engine-backed flows ----------------------------------------------------


@needs_engine
def test_threats_finds_the_scholars_mate_threat():
    session = _session(_SCHOLARS)
    with Engine() as engine:
        tools.bind(session, engine)
        session.goto_move(5)  # after 3.Qh5 — Black to move
        result = tools.threats()
    assert result["ok"] is True
    assert result["threatening_side"] == "white"
    threat = result["threat"]
    assert threat["move"] == "Qxf7#"
    assert threat["threatens_mate"] is True and threat["serious"] is True
    assert threat["size_centipawns"] is None  # no fake 100000-centipawn "size" for a mate
    assert session.current_ply == 5 and session.branch_stack == []  # board untouched


@needs_engine
def test_threats_reports_no_serious_threat_at_the_start():
    session = _session(_SCHOLARS)
    with Engine() as engine:
        tools.bind(session, engine)
        session.goto_move(0)
        result = tools.threats()
    assert result["ok"] is True
    assert result["threat"]["serious"] is False
    assert result["threat"]["size_centipawns"] is not None


@needs_engine
def test_threats_refuses_while_in_check():
    session = _session("1. e4 f5 2. Qh5+ *")
    with Engine() as engine:
        tools.bind(session, engine)
        session.goto_move(3)  # Black is in check
        result = tools.threats()
    assert result["ok"] is False and "in check" in result["error"]


@needs_engine
def test_try_line_walks_to_mate_without_moving_the_board():
    session = _session(_SCHOLARS)
    with Engine() as engine:
        tools.bind(session, engine)
        session.goto_move(6)  # after 3...Nf6?? — White to move
        before_fen = session.current.fen()
        result = tools.try_line("4.Qxf7#")
    assert result["ok"] is True
    assert result["line"]["moves"][0]["checkmate"] is True
    assert result["engine_after_line"]["outcome"] == "checkmate — white wins"
    assert session.current.fen() == before_fen and session.current_ply == 6


@needs_engine
def test_square_info_finds_the_weak_f7_square():
    session = _session(_SCHOLARS)
    with Engine() as engine:
        tools.bind(session, engine)
        session.goto_move(5)
        result = tools.square_info("f7")
    assert result["ok"] is True
    assert sorted(_squares(result["attacked_by"])) == ["c4", "h5"]
    assert _squares(result["defended_by"]) == ["e8"]  # only the king guards it


@needs_engine
def test_consequence_keeps_every_engine_reply_ply():
    # Regression: the consequence used to be capped at 8 plies while the engine's
    # reply line can be 8 by itself, so the final recapture (…Qxd8+ Kxd8) was cut
    # and half a queen trade read as "+11 material". Now every reply ply is kept.
    pgn = (
        '[SetUp "1"]\n'
        '[FEN "r1bqkb1r/ppp2ppp/3p1n2/4n3/4PP2/2N2N2/PPP3PP/R1BQKB1R b KQkq - 0 5"]\n\n'
        "5... h6 6. fxe5 *"
    )
    session = _session(pgn)
    with Engine() as engine:
        tools.bind(session, engine)
        result = tools.explain_move("5...h6")
    reply_line = result["engine_after"]["best_moves"][0]["line"]
    reply_plies = [t for t in reply_line.split() if not t.rstrip(".").isdigit()]
    consequence = result["consequence"]
    assert len(consequence["moves"]) == 1 + len(reply_plies)
    assert consequence["moves"][0]["move"] == "h6"


def test_probes_are_registered_for_gemini():
    for probe in (tools.square_info, tools.try_line, tools.threats):
        assert probe in tools.ALL_TOOLS
