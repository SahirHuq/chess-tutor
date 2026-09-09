"""Tests for the positional-feature layer (must be exactly correct — a wrong
positional 'fact' is worse than a vague one)."""

import chess

from features import (
    backward_pawns,
    center_control,
    connected_pawns,
    developed_minors,
    doubled_pawn_files,
    has_bishop_pair,
    is_trapped,
    isolated_pawns,
    king_safety,
    knight_outposts,
    mobility,
    passed_pawns,
    positional_changes,
    positional_features,
    rooks_on_open_files,
    trapped_pieces,
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


def test_positional_changes_flags_lost_king_shield_pawn():
    # White king g1 with g2/h2 shield; Black rook captures the h2 shield pawn while
    # the white king stays on g1 — a concrete loss of king cover.
    before = chess.Board("6kr/8/8/8/8/8/6PP/6K1 b - - 0 1")  # black rook h8, white g2/h2 shield
    after = before.copy()
    after.push_san("Rxh2")  # rook takes the h2 shield pawn; white king still on g1
    changes = positional_changes(before, after)
    assert "white's king lost a pawn from its shield" in changes


def test_positional_changes_flags_lost_center_control():
    before = chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")   # white pawn e4 grips d5/e4
    after = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")    # pawn back on e2, grip gone
    assert "white gave up control of the center" in positional_changes(before, after)


def test_positional_changes_flags_drop_in_activity():
    before = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")    # rook a1 roams freely
    after = chess.Board("4k3/8/8/8/8/8/rr6/R3K3 w - - 0 1")   # black rooks box it in
    changes = positional_changes(before, after)
    assert any("white's pieces have fewer active moves" in c for c in changes)


def test_rooks_on_open_and_semi_open_files():
    # White rook d1 on a fully open d-file; e-file has a black pawn only (semi-open for white).
    board = chess.Board("4k3/4p3/8/8/8/8/8/3RK3 w - - 0 1")
    rof = rooks_on_open_files(board, chess.WHITE)
    assert rof == {"d": "open"}
    board2 = chess.Board("4k3/4p3/8/8/8/8/8/4RK2 w - - 0 1")  # rook e1, black pawn e7
    assert rooks_on_open_files(board2, chess.WHITE) == {"e": "semi-open"}


def test_rook_blocked_by_own_pawn_is_not_open():
    board = chess.Board("4k3/8/8/8/8/8/3P4/3RK3 w - - 0 1")  # own pawn d2 in front of Rd1
    assert rooks_on_open_files(board, chess.WHITE) == {}


def test_knight_outpost_detected_and_hole_required():
    # White knight d5, defended by c4 pawn, no black pawn on c/e files to challenge -> outpost.
    board = chess.Board("4k3/8/8/3N4/2P5/8/8/4K3 w - - 0 1")
    assert "d5" in knight_outposts(board, chess.WHITE)
    # Add a black pawn on e6 that can advance to challenge d5 -> no longer an outpost.
    challenged = chess.Board("4k3/8/4p3/3N4/2P5/8/8/4K3 w - - 0 1")
    assert "d5" not in knight_outposts(challenged, chess.WHITE)


def test_positional_changes_flags_rook_taking_open_file():
    before = chess.Board("3rk3/8/8/8/8/8/8/4K2R w - - 0 1")  # white Rh1 (h-file open), black Rd8
    after = chess.Board("3rk3/8/8/8/8/8/8/3RK3 w - - 0 1")   # white rook now on the open d-file
    assert any("rook now controls the open d-file" in c for c in positional_changes(before, after))


def test_connected_pawns_phalanx_and_chain():
    phalanx = chess.Board("4k3/8/8/8/3PP3/8/8/4K3 w - - 0 1")  # d4,e4 side by side
    assert set(connected_pawns(phalanx, chess.WHITE)) == {"d4", "e4"}
    chain = chess.Board("4k3/8/8/4P3/3P4/8/8/4K3 w - - 0 1")   # d4 defends e5
    assert set(connected_pawns(chain, chess.WHITE)) == {"d4", "e5"}
    lone = chess.Board("4k3/8/8/8/P7/8/8/4K3 w - - 0 1")       # a4 alone
    assert connected_pawns(lone, chess.WHITE) == []


def test_backward_pawn_detected_but_not_isolated_or_supported():
    # White d3 pawn: its only neighbour (c4) has advanced past it so can't defend it,
    # and a black pawn on e5 covers d4 (the stop square) -> d3 is backward.
    board = chess.Board("4k3/8/8/4p3/2P5/3P4/8/4K3 w - - 0 1")
    assert "d3" in backward_pawns(board, chess.WHITE)
    # If the neighbour is still back on c2 it can defend d3's advance -> not backward.
    supported = chess.Board("4k3/8/8/4p3/8/3P4/2P5/4K3 w - - 0 1")
    assert "d3" not in backward_pawns(supported, chess.WHITE)
    # A truly lone pawn is isolated, not backward.
    lone = chess.Board("4k3/8/8/4p3/8/3P4/8/4K3 w - - 0 1")
    assert backward_pawns(lone, chess.WHITE) == []


def test_trapped_bishop_with_no_safe_square():
    # White Bh4 is attacked by the g5 pawn; both escapes (Bxg5, Bxg3) land on squares
    # defended by black f-pawns, so the bishop is trapped.
    board = chess.Board("4k3/8/5p2/6p1/5p1B/6p1/8/K7 w - - 0 1")
    assert is_trapped(board, chess.H4)
    assert "h4" in trapped_pieces(board, chess.WHITE)


def test_trapped_not_claimed_for_side_not_to_move():
    # Same trapped geometry, but it is BLACK to move: we must not assess the white
    # bishop (can't generate its moves), so we never claim it is trapped.
    board = chess.Board("4k3/8/5p2/6p1/5p1B/6p1/8/K7 b - - 0 1")
    assert is_trapped(board, chess.H4) is False
    assert trapped_pieces(board, chess.WHITE) == []


def test_piece_with_a_safe_escape_is_not_trapped():
    board = chess.Board("4k3/8/8/6p1/7B/8/8/4K3 w - - 0 1")  # Bh4 attacked by g5, but free
    assert is_trapped(board, chess.H4) is False


def test_positional_features_snapshot_is_complete_and_both_sided():
    # The snapshot the tutor reasons over must carry every grounded factor, for both
    # sides, so open-ended questions aren't blind to whole categories.
    feats = positional_features(chess.Board())
    expected = {
        "bishop_pair", "doubled_pawns", "isolated_pawns", "backward_pawns",
        "connected_pawns", "passed_pawns", "rooks_on_open_files", "knight_outposts",
        "trapped_pieces", "king_safety", "mobility", "developed_minors", "center_control",
    }
    assert set(feats) == expected
    for key, value in feats.items():
        assert "white" in value and "black" in value, key
    # Spot-check a couple of values at the start position.
    assert feats["bishop_pair"] == {"white": True, "black": True}
    assert len(feats["connected_pawns"]["white"]) == 8  # the start rank is one big phalanx


def test_lost_shield_not_flagged_when_king_moves():
    # The king itself relocates (g1 -> f1): the shield naturally differs, but that is
    # a king move, not a weakening of a fixed king — must NOT be reported.
    before = chess.Board("6k1/8/8/8/8/8/6PP/6K1 w - - 0 1")
    after = before.copy()
    after.push_san("Kf1")
    assert "white's king lost a pawn from its shield" not in positional_changes(before, after)
