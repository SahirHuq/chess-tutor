"""Position facts — the *grounding* half of the brain.

Pure functions that turn a `chess.Board` into plain, verifiable facts: material
balance, hanging pieces, checks, castling rights. These are the evidence the
model narrates, so the engine's bare "+1.5" becomes "+1.5 because your knight
on e5 is undefended." No engine and no model here — trivially unit-testable.
"""

from __future__ import annotations

from typing import Optional

import chess

# Standard relative piece values (king excluded — it can't be captured).
PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def _color_name(color: chess.Color) -> str:
    return "white" if color else "black"


def material_balance(board: chess.Board) -> dict:
    """Total non-king material per side, and the difference (White minus Black)."""
    white = black = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            white += value
        else:
            black += value
    return {"white": white, "black": black, "diff": white - black}


def hanging_pieces(board: chess.Board) -> list[dict]:
    """Pieces that are attacked by the enemy and have **no** defender.

    We deliberately report only *attacked-and-undefended* pieces — that claim is
    always true. We avoid 'attackers > defenders' guesses, which depend on
    capture order and piece values and could mislead.
    """
    out: list[dict] = []
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        attackers = board.attackers(not piece.color, square)
        defenders = board.attackers(piece.color, square)
        if attackers and not defenders:
            out.append(
                {
                    "square": chess.square_name(square),
                    "piece": chess.piece_name(piece.piece_type),
                    "color": _color_name(piece.color),
                    "attacked_by": len(attackers),
                    "defended_by": len(defenders),
                }
            )
    return out


def checkers(board: chess.Board) -> list[dict]:
    """Enemy pieces currently giving check to the side to move (empty if none)."""
    if not board.is_check():
        return []
    out: list[dict] = []
    for square in board.checkers():
        piece = board.piece_at(square)
        assert piece is not None
        out.append(
            {
                "square": chess.square_name(square),
                "piece": chess.piece_name(piece.piece_type),
                "color": _color_name(piece.color),
            }
        )
    return out


def castling_rights(board: chess.Board) -> dict:
    return {
        "white_kingside": board.has_kingside_castling_rights(chess.WHITE),
        "white_queenside": board.has_queenside_castling_rights(chess.WHITE),
        "black_kingside": board.has_kingside_castling_rights(chess.BLACK),
        "black_queenside": board.has_queenside_castling_rights(chess.BLACK),
    }


# ---------------------------------------------------------------------------
# Positional features — grounded "why" for non-tactical moves. Every function
# reports something verifiably true about the position (no judgement calls), so
# the model can explain a positional move without inventing reasons.
# ---------------------------------------------------------------------------


def _pawn_files(board: chess.Board, color: chess.Color) -> list[int]:
    return [chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)]


def has_bishop_pair(board: chess.Board, color: chess.Color) -> bool:
    """True if `color` has two or more bishops (the classic 'bishop pair')."""
    return len(board.pieces(chess.BISHOP, color)) >= 2


def doubled_pawn_files(board: chess.Board, color: chess.Color) -> list[str]:
    """File letters (a–h) on which `color` has two or more pawns."""
    files = _pawn_files(board, color)
    return [chr(ord("a") + f) for f in sorted(set(files)) if files.count(f) > 1]


def isolated_pawns(board: chess.Board, color: chess.Color) -> list[str]:
    """Squares of `color`'s pawns with no friendly pawn on an adjacent file."""
    files = set(_pawn_files(board, color))
    out = []
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        if f - 1 not in files and f + 1 not in files:
            out.append(chess.square_name(sq))
    return sorted(out)


def passed_pawns(board: chess.Board, color: chess.Color) -> list[str]:
    """Squares of `color`'s passed pawns (no enemy pawn ahead on same/adjacent files)."""
    enemy_pawns = list(board.pieces(chess.PAWN, not color))
    out = []
    for sq in board.pieces(chess.PAWN, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ahead = lambda er: er > r if color == chess.WHITE else er < r  # noqa: E731
        if not any(
            abs(chess.square_file(e) - f) <= 1 and ahead(chess.square_rank(e))
            for e in enemy_pawns
        ):
            out.append(chess.square_name(sq))
    return sorted(out)


_MINOR_HOME = {
    chess.WHITE: {chess.B1, chess.G1, chess.C1, chess.F1},
    chess.BLACK: {chess.B8, chess.G8, chess.C8, chess.F8},
}
_CENTER = (chess.D4, chess.E4, chess.D5, chess.E5)


def developed_minors(board: chess.Board, color: chess.Color) -> int:
    """How many of `color`'s knights and bishops have left their starting squares."""
    home = _MINOR_HOME[color]
    minors = list(board.pieces(chess.KNIGHT, color)) + list(board.pieces(chess.BISHOP, color))
    return sum(1 for sq in minors if sq not in home)


def center_control(board: chess.Board, color: chess.Color) -> int:
    """`color`'s grip on the four central squares (d4/e4/d5/e5): each square
    counts once if `color` occupies it, plus once per attacker on it."""
    score = 0
    for sq in _CENTER:
        piece = board.piece_at(sq)
        if piece is not None and piece.color == color:
            score += 1
        score += len(board.attackers(color, sq))
    return score


def king_safety(board: chess.Board, color: chess.Color) -> dict:
    """A rough, factual read on `color`'s king: where it is, its pawn shield,
    and how many squares around it the enemy attacks."""
    ksq = board.king(color)
    if ksq is None:
        return {"king_square": None, "shield_pawns": 0, "squares_attacked_near_king": 0}
    enemy = not color
    zone = [ksq] + [s for s in chess.SQUARES if chess.square_distance(s, ksq) == 1]
    attacked = sum(1 for s in zone if board.attackers(enemy, s))
    kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
    step = 1 if color == chess.WHITE else -1
    shield = 0
    for df in (-1, 0, 1):
        for dr in (1, 2):
            f, r = kf + df, kr + dr * step
            if 0 <= f < 8 and 0 <= r < 8:
                piece = board.piece_at(chess.square(f, r))
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    shield += 1
    return {
        "king_square": chess.square_name(ksq),
        "shield_pawns": shield,
        "squares_attacked_near_king": attacked,
    }


def mobility(board: chess.Board) -> dict:
    """Legal-move counts per side — a rough proxy for piece activity."""
    side_to_move = board.legal_moves.count()
    other: Optional[int] = None
    if not board.is_check():  # a null move is illegal while in check
        probe = board.copy(stack=False)
        probe.push(chess.Move.null())
        other = probe.legal_moves.count()
    if board.turn == chess.WHITE:
        return {"white": side_to_move, "black": other}
    return {"white": other, "black": side_to_move}


def positional_features(board: chess.Board) -> dict:
    """All positional facts for a position, as a plain JSON-able dict."""
    return {
        "bishop_pair": {
            "white": has_bishop_pair(board, chess.WHITE),
            "black": has_bishop_pair(board, chess.BLACK),
        },
        "doubled_pawns": {
            "white": doubled_pawn_files(board, chess.WHITE),
            "black": doubled_pawn_files(board, chess.BLACK),
        },
        "isolated_pawns": {
            "white": isolated_pawns(board, chess.WHITE),
            "black": isolated_pawns(board, chess.BLACK),
        },
        "passed_pawns": {
            "white": passed_pawns(board, chess.WHITE),
            "black": passed_pawns(board, chess.BLACK),
        },
        "king_safety": {
            "white": king_safety(board, chess.WHITE),
            "black": king_safety(board, chess.BLACK),
        },
        "mobility": mobility(board),
        "developed_minors": {
            "white": developed_minors(board, chess.WHITE),
            "black": developed_minors(board, chess.BLACK),
        },
        "center_control": {
            "white": center_control(board, chess.WHITE),
            "black": center_control(board, chess.BLACK),
        },
    }


def positional_changes(before: chess.Board, after: chess.Board) -> list[str]:
    """Plain-English list of the positional things a move changed (before -> after).

    This is the grounded 'why' for a positional move — only true, observed
    differences, never a judgement the engine didn't make.
    """
    changes: list[str] = []
    for color in (chess.WHITE, chess.BLACK):
        name = _color_name(color)
        if has_bishop_pair(before, color) and not has_bishop_pair(after, color):
            changes.append(f"{name} gave up the bishop pair")
        new_doubled = set(doubled_pawn_files(after, color)) - set(
            doubled_pawn_files(before, color)
        )
        for f in sorted(new_doubled):
            changes.append(f"{name} now has doubled pawns on the {f}-file")
        new_iso = set(isolated_pawns(after, color)) - set(isolated_pawns(before, color))
        for sq in sorted(new_iso):
            changes.append(f"{name} now has an isolated pawn on {sq}")
        new_passed = set(passed_pawns(after, color)) - set(passed_pawns(before, color))
        for sq in sorted(new_passed):
            changes.append(f"{name} now has a passed pawn on {sq}")
        ks_before = king_safety(before, color)["squares_attacked_near_king"]
        ks_after = king_safety(after, color)["squares_attacked_near_king"]
        if ks_after - ks_before >= 2:
            changes.append(f"{name}'s king is more exposed (more squares attacked near it)")
        if developed_minors(after, color) > developed_minors(before, color):
            changes.append(f"{name} developed a piece")
        if center_control(after, color) - center_control(before, color) >= 2:
            changes.append(f"{name} gained more control of the center")
    return changes


def _attacked_pieces(board: chess.Board, square: int) -> list[str]:
    """Enemy minor/major pieces (not pawns, not the king) attacked by the piece
    now standing on `square`. This is the move's concrete point ('attacks the
    bishop on g4') — pawns are excluded as noise, the king is a check (flagged
    separately)."""
    piece = board.piece_at(square)
    if piece is None:
        return []
    out = []
    for target_sq in board.attacks(square):
        target = board.piece_at(target_sq)
        if (
            target is not None
            and target.color != piece.color
            and target.piece_type not in (chess.PAWN, chess.KING)
        ):
            out.append(f"{chess.piece_name(target.piece_type)} on {chess.square_name(target_sq)}")
    return sorted(out)


def annotate_line(board: chess.Board, moves: list[chess.Move], max_plies: int = 6) -> dict:
    """Walk a line and report the *mechanism*: what each move captures, attacks,
    and whether it checks, plus the net material change. This is what lets the
    model say "h3 attacks the bishop on g4" or "Bxd1 wins the queen" precisely,
    instead of guessing the point of a move.
    """
    work = board.copy()
    before = material_balance(work)
    annotated: list[dict] = []
    for move in moves[:max_plies]:
        captured = None
        if work.is_capture(move):
            if work.is_en_passant(move):
                captured = "pawn"
            else:
                target = work.piece_at(move.to_square)
                captured = chess.piece_name(target.piece_type) if target else None
        mover = "white" if work.turn else "black"  # whose move (before pushing)
        from_square = chess.square_name(move.from_square)
        san = work.san(move)  # SAN must be computed before the move is pushed
        work.push(move)
        annotated.append(
            {
                "move": san,
                "side": mover,
                "from": from_square,
                "captures": captured,
                "attacks": _attacked_pieces(work, move.to_square),
                "gives_check": work.is_check(),
            }
        )
    after = material_balance(work)
    swing = after["diff"] - before["diff"]  # (White - Black); negative = Black gained

    captures = [m for m in annotated if m["captures"]]
    if captures:
        sequence = ", then ".join(
            f"{m['move']} (captures the {m['captures']})" for m in captures
        )
    else:
        sequence = "no captures"
    if swing <= -3:
        outcome = f"Black ends up about {abs(swing)} points of material ahead"
    elif swing >= 3:
        outcome = f"White ends up about {swing} points of material ahead"
    else:
        outcome = "material stays even (an equal trade)"
    # A ready-to-relay, factual sentence so the model needn't infer any geometry.
    summary = f"The line goes: {sequence}. Net material: {outcome}."

    return {
        "summary": summary,
        "moves": annotated,
        "material_before": before,
        "material_after": after,
        "material_swing_white_minus_black": swing,
    }


def position_facts(board: chess.Board) -> dict:
    """All grounding facts for a position, as a plain JSON-able dict."""
    return {
        "fen": board.fen(),
        "side_to_move": _color_name(board.turn),
        "fullmove_number": board.fullmove_number,
        "in_check": board.is_check(),
        "checkers": checkers(board),
        "material": material_balance(board),
        "hanging_pieces": hanging_pieces(board),
        "castling_rights": castling_rights(board),
    }
