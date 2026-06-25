"""Tests for the positional-feature layer (must be exactly correct — a wrong
positional 'fact' is worse than a vague one)."""

import chess

from features import (
    center_control,
    developed_minors,
    doubled_pawn_files,
    has_bishop_pair,
    isolated_pawns,
    king_safety,
    mobility,
    passed_pawns,
    positional_changes,
)

START = chess.Board()


def test_bishop_pair_start():
    assert has_bishop_pair(START, chess.WHITE) is True
    assert has_bishop_pair(START, chess.BLACK) is True


def test_bishop_pair_lost():
    board = chess.Board("k7/8/8/8/8/8/8/K1B5 w - - 0 1")  # only one white bishop
    assert has_bishop_pair(board, chess.WHITE) is False


def test_doubled_pawns():
    board = chess.Board("k7/8/8/8/2P5/2P5/8/K7 w - - 0 1")  # white pawns c4, c3
    assert doubled_pawn_files(board, chess.WHITE) == ["c"]
    assert doubled_pawn_files(board, chess.BLACK) == []


def test_isolated_pawn():
    board = chess.Board("k7/8/8/8/P7/8/8/K7 w - - 0 1")  # lone a4 pawn
    assert isolated_pawns(board, chess.WHITE) == ["a4"]


def test_not_isolated_with_neighbor():
    board = chess.Board("k7/8/8/8/PP6/8/8/K7 w - - 0 1")  # a4 and b4 support each other
    assert isolated_pawns(board, chess.WHITE) == []


def test_passed_pawn():
    board = chess.Board("k7/8/8/4P3/8/8/8/K7 w - - 0 1")  # e5, no black pawns
    assert passed_pawns(board, chess.WHITE) == ["e5"]


def test_not_passed_when_blocked_ahead():
    board = chess.Board("k7/3p4/8/4P3/8/8/8/K7 w - - 0 1")  # black d7 controls in front
    assert passed_pawns(board, chess.WHITE) == []


def test_king_safety_start():
    ks = king_safety(START, chess.WHITE)
    assert ks["king_square"] == "e1"
    assert ks["shield_pawns"] == 3  # d2, e2, f2 in front of the king
    assert ks["squares_attacked_near_king"] == 0


def test_mobility_start():
    mob = mobility(START)
    assert mob["white"] == 20
    assert mob["black"] == 20


def test_development_count():
    assert developed_minors(START, chess.WHITE) == 0  # all minors home at the start
    after_nf3 = chess.Board()
    after_nf3.push_san("Nf3")
    assert developed_minors(after_nf3, chess.WHITE) == 1


def test_center_control_increases_with_e4():
    before = center_control(START, chess.WHITE)
    after = chess.Board()
    after.push_san("e4")
    assert center_control(after, chess.WHITE) > before


def test_positional_changes_development_and_center():
    before = chess.Board()
    after = chess.Board()
    after.push_san("e4")
    changes = positional_changes(before, after)
    assert "white gained more control of the center" in changes


def test_annotate_line_reports_attacks():
    from features import annotate_line

    # White pawn h2, Black bishop on g4. h3 attacks that bishop.
    board = chess.Board("7k/8/8/8/6b1/8/7P/7K w - - 0 1")
    line = annotate_line(board, [board.parse_san("h3")])
    first = line["moves"][0]
    assert first["move"] == "h3"
    assert first["side"] == "white"
    assert first["from"] == "h2"
    assert first["captures"] is None
    assert first["attacks"] == ["bishop on g4"]


def test_positional_changes_bishop_pair_and_doubled():
    before = chess.Board("k7/5pp1/8/8/8/8/5PP1/K1B2B2 w - - 0 1")  # 2 white bishops; f7,g7
    after = chess.Board("k7/5p2/5p2/8/8/8/5PP1/K1B5 w - - 0 1")  # 1 bishop; f7,f6 doubled
    changes = positional_changes(before, after)
    assert "white gave up the bishop pair" in changes
    assert "black now has doubled pawns on the f-file" in changes
