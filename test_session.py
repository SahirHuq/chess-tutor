"""Tests for session navigation + safe what-if branching."""

import io

import chess.pgn
import pytest

from session import SessionState


def _game(pgn: str) -> chess.pgn.Game:
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    return game

# Fool's mate: 1.f3 e5 2.g4 Qh4#  (4 half-moves)
PGN = "1. f3 e5 2. g4 Qh4# 0-1"


def make_session() -> SessionState:
    game = chess.pgn.read_game(io.StringIO(PGN))
    assert game is not None
    return SessionState.from_game(game)


def test_mainline_is_parsed():
    s = make_session()
    assert s.mainline_san == ["f3", "e5", "g4", "Qh4#"]
    assert len(s.mainline_moves) == 4


def test_goto_move_navigates_and_orients():
    s = make_session()
    start = s.goto_move(0)
    assert start["on_mainline"] is True
    assert start["last_move"] is None
    assert start["next_mainline_move"] == "f3"

    mid = s.goto_move(2)  # after 1.f3 e5
    assert mid["last_move"] == "e5"
    assert mid["next_mainline_move"] == "g4"
    assert mid["side_to_move"] == "white"


def test_goto_move_out_of_range_is_rejected():
    s = make_session()
    res = s.goto_move(99)
    assert res["ok"] is False
    assert "out of range" in res["error"]


def test_play_move_branches_without_corrupting_mainline():
    s = make_session()
    s.goto_move(2)  # after 1.f3 e5, white to move
    branch = s.play_move("g4")
    assert branch["ok"] is True
    assert branch["on_mainline"] is False
    assert branch["branch_depth"] == 1
    assert branch["last_move"] == "g4"
    # The real game is untouched.
    assert s.mainline_san == ["f3", "e5", "g4", "Qh4#"]
    assert len(s.mainline_moves) == 4


def test_go_back_restores_previous_position():
    s = make_session()
    s.goto_move(2)
    s.play_move("g4")
    back = s.go_back()
    assert back["ok"] is True
    assert back["on_mainline"] is True
    assert back["branch_depth"] == 0
    assert back["next_mainline_move"] == "g4"


def test_go_back_at_base_is_handled():
    s = make_session()
    s.goto_move(2)
    res = s.go_back()
    assert res["ok"] is False
    assert "nothing to go back" in res["error"]


def test_illegal_move_is_rejected_with_reason():
    s = make_session()
    s.goto_move(0)
    res = s.play_move("Qh4")  # not legal from the starting position
    assert res["ok"] is False
    assert res["error"].startswith("illegal:")


def test_resolve_ply_disambiguates_repeated_san():
    import tools

    # A game where Nf3 (moves 1 & 4), Nc6 (5 & 15), O-O (W move 9 & B move 10) repeat.
    pgn = (
        "1. Nf3 d5 2. g3 e5 3. Nxe5 f6 4. Nf3 Bc5 5. e3 Nc6 6. d4 Bb6 7. Bb5 a6 "
        "8. Bxc6+ bxc6 9. O-O Ne7 10. c3 O-O 11. a4 Bh3 12. Re1 c5 13. a5 Ba7 "
        "14. b4 cxb4 15. cxb4 Nc6 16. Ba3 Qe7 17. b5 1-0"
    )
    s = SessionState.from_game(chess.pgn.read_game(io.StringIO(pgn)))

    assert tools._resolve_ply(s, "1.Nf3") == 1
    assert tools._resolve_ply(s, "4.Nf3") == 7  # the SECOND Nf3, not the first
    assert tools._resolve_ply(s, "5...Nc6") == 10
    assert tools._resolve_ply(s, "15...Nc6") == 30  # the second Nc6
    assert tools._resolve_ply(s, "9.O-O") == 17  # White's castling
    assert tools._resolve_ply(s, "10...O-O") == 20  # Black's castling (not White's)
    assert tools._resolve_ply(s, "Nf3") == 1  # bare SAN still falls back to first


def test_best_forcing_mate_is_not_marked_unclear():
    import tools

    verdict = tools._classify_move(
        mover="black",
        best_value=-100000,
        actual_value=None,  # game-over positions have no next engine best move
        played_the_best=True,
    )

    assert verdict == {"verdict": "best move", "centipawns_lost": 0}


def test_nested_what_if_then_unwind():
    s = make_session()
    s.goto_move(0)
    s.play_move("e4")  # branch 1
    s.play_move("e5")  # branch 2 (what-if inside a what-if)
    assert s.orientation()["branch_depth"] == 2
    s.go_back()
    s.go_back()
    assert s.orientation()["branch_depth"] == 0
    assert s.orientation()["on_mainline"] is True
    # Mainline still pristine.
    assert s.mainline_san == ["f3", "e5", "g4", "Qh4#"]


# ---- player identity (names, result, which side the user played) ----------


_NAMED_PGN = '[White "Alice"]\n[Black "Bob"]\n[Result "0-1"]\n\n1. f3 e5 2. g4 Qh4# 0-1'


def test_pgn_player_names_and_result_are_preserved():
    s = SessionState.from_game(_game(_NAMED_PGN))
    assert s.white_player == "Alice"
    assert s.black_player == "Bob"
    assert s.result == "0-1"
    assert s.display_white() == "Alice"
    assert s.display_black() == "Bob"


def test_user_can_be_white_or_black():
    as_white = SessionState.from_game(_game(_NAMED_PGN), player_color="white")
    assert as_white.player_color == "white"
    assert as_white.opponent_color() == "black"

    as_black = SessionState.from_game(_game(_NAMED_PGN), player_color="black")
    assert as_black.player_color == "black"
    assert as_black.opponent_color() == "white"


def test_player_color_defaults_to_unknown():
    s = SessionState.from_game(_game(_NAMED_PGN))
    assert s.player_color is None
    assert s.opponent_color() is None


def test_set_player_color_can_be_cleared_and_validates():
    s = SessionState.from_game(_game(_NAMED_PGN), player_color="white")
    s.set_player_color(None)
    assert s.player_color is None
    with pytest.raises(ValueError):
        s.set_player_color("grey")


def test_from_game_rejects_invalid_player_color():
    with pytest.raises(ValueError):
        SessionState.from_game(_game(_NAMED_PGN), player_color="grey")


def test_missing_pgn_names_still_work():
    # No White/Black headers — python-chess fills "?"; display falls back cleanly.
    s = SessionState.from_game(_game("1. f3 e5 2. g4 Qh4# 0-1"))
    assert s.white_player == "?"
    assert s.black_player == "?"
    assert s.display_white() == "White"
    assert s.display_black() == "Black"
    # The game itself still loads fully.
    assert s.mainline_san == ["f3", "e5", "g4", "Qh4#"]
