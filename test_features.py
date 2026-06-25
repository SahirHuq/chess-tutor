"""Unit tests for the grounding layer (no engine, no model needed)."""

import chess

from features import (
    castling_rights,
    checkers,
    hanging_pieces,
    material_balance,
    position_facts,
)

START = chess.Board()


def test_start_material_is_equal():
    mat = material_balance(START)
    assert mat["white"] == mat["black"] == 39  # 8 + 2*3 + 2*3 + 2*5 + 9
    assert mat["diff"] == 0


def test_start_has_no_hanging_pieces_and_no_check():
    assert hanging_pieces(START) == []
    assert checkers(START) == []
    assert position_facts(START)["in_check"] is False


def test_start_full_castling_rights():
    rights = castling_rights(START)
    assert all(rights.values())


def test_hanging_knight_is_detected():
    # White knight on e6 is attacked by the black pawn on d7 and undefended.
    board = chess.Board("4k3/3p4/4N3/8/8/8/8/4K3 w - - 0 1")
    hanging = hanging_pieces(board)
    assert len(hanging) == 1
    piece = hanging[0]
    assert piece["square"] == "e6"
    assert piece["piece"] == "knight"
    assert piece["color"] == "white"
    assert piece["attacked_by"] == 1
    assert piece["defended_by"] == 0


def test_defended_piece_is_not_hanging():
    # Same knight on e6, now defended by a white pawn on d5 → not hanging.
    board = chess.Board("4k3/3p4/4N3/3P4/8/8/8/4K3 w - - 0 1")
    assert hanging_pieces(board) == []


def test_check_is_reported_with_the_checking_piece():
    # Black queen on e8 checks the white king down the open e-file.
    board = chess.Board("k3q3/8/8/8/8/8/8/4K3 w - - 0 1")
    facts = position_facts(board)
    assert facts["in_check"] is True
    assert len(facts["checkers"]) == 1
    assert facts["checkers"][0]["square"] == "e8"
    assert facts["checkers"][0]["piece"] == "queen"


def test_material_difference_counts_extra_material():
    # White knight (3) vs black pawn (1) → White up 2.
    board = chess.Board("4k3/3p4/4N3/8/8/8/8/4K3 w - - 0 1")
    mat = material_balance(board)
    assert mat["white"] == 3
    assert mat["black"] == 1
    assert mat["diff"] == 2
