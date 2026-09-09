"""Tests for tactical-motif detection (must be exactly correct — a fabricated
'fork' is worse than no fork at all)."""

import chess

from tactics import move_tactics


def _tactics(fen: str, san: str) -> list[str]:
    board = chess.Board(fen)
    return move_tactics(board, board.parse_san(san))


def test_knight_fork_of_king_and_rook_is_reported():
    # White knight e5 -> c6 forks the black king (b8) and rook (a7) — a real fork.
    out = _tactics("1k5r/r7/8/4N3/8/8/8/4K3 w - - 0 1", "Nc6+")
    assert any("forks" in t and "king" in t and "rook on a7" in t for t in out)


def test_pawn_fork_of_two_minors_is_reported():
    # White pawn e4 -> e5? no; use a classic: pawn d5 forks knights on c6 and e6.
    out = _tactics("4k3/8/2n1n3/3P4/8/8/8/4K3 w - - 0 1", "d6")  # d5-d6 hits c7? rebuild
    # Rebuild cleanly: pawn on e4 to e5 forking knights on d6 and f6.
    out = _tactics("4k3/8/3n1n2/8/4P3/8/8/4K3 w - - 0 1", "e5")
    assert any("forks" in t and "d6" in t and "f6" in t for t in out)


def test_high_value_piece_brushing_defended_pieces_is_not_a_fork():
    # White queen attacks two pawns — NOT a fork (queen is worth more than pawns).
    out = _tactics("4k3/8/8/3p1p2/8/8/8/Q3K3 w - - 0 1", "Qd4")
    assert not any("forks" in t for t in out)


def test_pin_against_king_is_reported():
    # Rook h1 -> e1 puts the black knight e4 in front of the black king e8: an
    # absolute pin (white king on g3, off the e-file, so the rook's path is clear).
    out = _tactics("4k3/8/8/8/4n3/K7/8/7R w - - 0 1", "Re1")
    assert any("pins the knight on e4 against the king" in t for t in out)


def test_move_that_hangs_own_piece_is_reported():
    # Bishop c1 -> d2 walks onto the d-file where the black rook d8 attacks it, with
    # no white defender (the white king is on h1, far away) -> hanging.
    out = _tactics("3r3k/8/8/8/8/8/8/2B4K w - - 0 1", "Bd2")
    assert any("leaves the bishop on d2 hanging" in t for t in out)


def test_quiet_safe_move_reports_no_tactics():
    out = _tactics("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "e4")
    assert out == []
