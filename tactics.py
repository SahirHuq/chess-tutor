"""Tactical motifs — the *sharp* half of the grounded 'why'.

`features.py` reports slow, structural facts (pawns, bishop pair, king cover).
This module reports the fast ones: the concrete tactics a move creates — a fork,
a pin, or a piece left hanging. Like everything in the brain, every claim here is
a verifiable geometric fact about the board (this piece attacks those two pieces;
this piece is pinned to its king; this piece is attacked and has no defender), so
the tutor can NAME the tactic without ever inventing one. We deliberately report
only motifs that are always true by construction — we do not guess "winning",
which depends on the follow-up the engine already scores elsewhere.
"""

from __future__ import annotations

import chess

from features import PIECE_VALUES, _color_name, hanging_pieces


def _piece_label(board: chess.Board, square: int) -> str:
    piece = board.piece_at(square)
    assert piece is not None
    return f"{chess.piece_name(piece.piece_type)} on {chess.square_name(square)}"


def _forks(after: chess.Board, square: int, mover: chess.Color) -> list[str]:
    """A genuine double attack by the piece now on `square`: it hits two or more
    targets that are each at least as valuable as itself (so the geometry actually
    wins material), or it checks the king AND hits a piece. A queen brushing two
    defended pawns is not reported — that filter keeps 'fork' honest."""
    piece = after.piece_at(square)
    if piece is None or piece.color != mover:
        return []
    forker_value = PIECE_VALUES.get(piece.piece_type, 0)
    targets: list[str] = []
    checks_king = False
    for target_sq in after.attacks(square):
        victim = after.piece_at(target_sq)
        if victim is None or victim.color == mover:
            continue
        if victim.piece_type == chess.KING:
            checks_king = True
        elif PIECE_VALUES.get(victim.piece_type, 0) >= forker_value:
            targets.append(_piece_label(after, target_sq))
    hits = targets + (["the king (a check)"] if checks_king else [])
    if len(hits) < 2:
        return []
    joined = " and ".join(hits)
    return [f"the {chess.piece_name(piece.piece_type)} on {chess.square_name(square)} "
            f"forks {joined}"]


def _pins_created(before: chess.Board, after: chess.Board, mover: chess.Color) -> list[str]:
    """Enemy pieces that the move pins against their own king (an absolute pin) and
    were not pinned before. python-chess decides the pin, so it is always true."""
    if after.is_check():  # a check dominates; don't also narrate pins this ply
        return []
    enemy = not mover
    out: list[str] = []
    for square, piece in after.piece_map().items():
        if piece.color != enemy or piece.piece_type == chess.KING:
            continue
        if after.is_pinned(enemy, square) and not before.is_pinned(enemy, square):
            out.append(f"pins the {chess.piece_name(piece.piece_type)} "
                       f"on {chess.square_name(square)} against the king")
    return out


def _new_hanging(before: chess.Board, after: chess.Board, mover: chess.Color) -> list[str]:
    """The mover's own pieces left attacked and undefended by the move — the classic
    'you just hung it'. Reuses the conservative `hanging_pieces` (attacked AND zero
    defenders), so a piece protected by a recapture is never falsely flagged."""
    mover_name = _color_name(mover)
    before_hanging = {h["square"] for h in hanging_pieces(before) if h["color"] == mover_name}
    out: list[str] = []
    for h in hanging_pieces(after):
        if h["color"] == mover_name and h["square"] not in before_hanging:
            out.append(f"leaves the {h['piece']} on {h['square']} hanging "
                       f"(attacked and undefended)")
    return out


def move_tactics(before: chess.Board, move: chess.Move) -> list[str]:
    """Named, verifiable tactics a single move creates, from the mover's side:
    forks and pins it sets up, and any of the mover's own pieces it leaves hanging.
    Empty when the move makes no concrete tactic — never a guess."""
    mover = before.turn
    after = before.copy()
    after.push(move)
    return (
        _forks(after, move.to_square, mover)
        + _pins_created(before, after, mover)
        + _new_hanging(before, after, mover)
    )
