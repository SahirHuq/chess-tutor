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


def _square_attacked_by_pawn(board: chess.Board, square: int, by_color: chess.Color) -> bool:
    """Would a `by_color` pawn capture on `square`? (Used for 'can't advance safely'.)"""
    return any(
        (p := board.piece_at(a)) is not None and p.piece_type == chess.PAWN
        for a in board.attackers(by_color, square)
    )


def connected_pawns(board: chess.Board, color: chess.Color) -> list[str]:
    """Pawns that are part of a healthy chain: a friendly pawn beside them (a phalanx)
    or one defending/defended diagonally. Connected pawns support each other and each
    other's advance — the opposite of the isolated/doubled/backward weaknesses."""
    pawn_set = set(board.pieces(chess.PAWN, color))
    out: list[str] = []
    for sq in pawn_set:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        neighbours = [
            chess.square(f + df, r + dr)
            for df in (-1, 1)
            for dr in (-1, 0, 1)
            if 0 <= f + df < 8 and 0 <= r + dr < 8
        ]
        if any(n in pawn_set for n in neighbours):
            out.append(chess.square_name(sq))
    return sorted(out)


def backward_pawns(board: chess.Board, color: chess.Color) -> list[str]:
    """Pawns left behind their neighbours: there ARE friendly pawns on adjacent files
    (so it isn't merely isolated), but all of them have advanced past it, so none can
    defend it — and the square in front is covered by an enemy pawn, so it can't
    advance safely either. A classic long-term weakness the engine penalises."""
    own = list(board.pieces(chess.PAWN, color))
    out: list[str] = []
    for sq in own:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        neighbours = [p for p in own if abs(chess.square_file(p) - f) == 1]
        if not neighbours:  # no neighbours at all -> isolated, not backward
            continue
        if color == chess.WHITE:
            if any(chess.square_rank(p) <= r for p in neighbours):  # one can still support it
                continue
            stop = chess.square(f, r + 1) if r + 1 <= 7 else None
        else:
            if any(chess.square_rank(p) >= r for p in neighbours):
                continue
            stop = chess.square(f, r - 1) if r - 1 >= 0 else None
        if stop is not None and _square_attacked_by_pawn(board, stop, not color):
            out.append(chess.square_name(sq))
    return sorted(out)


def is_trapped(board: chess.Board, square: int) -> bool:
    """Is the piece on `square` trapped — attacked by a cheaper enemy piece, with no
    move to a square where it isn't again attacked by something cheaper or a pawn?
    Only assessable for the side to move (it needs that piece's own moves), so we
    return False otherwise — never a guess. Conservative on purpose: when it fires,
    the piece really is losing material with nowhere to run."""
    piece = board.piece_at(square)
    if piece is None or piece.piece_type in (chess.PAWN, chess.KING):
        return False
    if board.turn != piece.color:  # can't generate this piece's moves when it isn't its turn
        return False
    value = PIECE_VALUES[piece.piece_type]
    attacked_by_cheaper = any(
        PIECE_VALUES.get((p := board.piece_at(a)) and p.piece_type, 99) < value
        for a in board.attackers(not piece.color, square)
    )
    if not attacked_by_cheaper:
        return False
    for move in board.legal_moves:
        if move.from_square != square:
            continue
        after = board.copy()
        after.push(move)
        dest = move.to_square
        cheaper_attacker = any(
            PIECE_VALUES.get((p := after.piece_at(a)) and p.piece_type, 99) < value
            for a in after.attackers(not piece.color, dest)
        )
        if not cheaper_attacker and not _square_attacked_by_pawn(after, dest, not piece.color):
            return False  # found a safe escape
    return True


def trapped_pieces(board: chess.Board, color: chess.Color) -> list[str]:
    """Squares of `color`'s trapped minor/major pieces (only when it's `color`'s turn)."""
    if board.turn != color:
        return []
    return sorted(
        chess.square_name(sq)
        for sq, piece in board.piece_map().items()
        if piece.color == color and is_trapped(board, sq)
    )


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


def rooks_on_open_files(board: chess.Board, color: chess.Color) -> dict[str, str]:
    """Files where `color` has a rook with no friendly pawn in the way, labelled
    'open' (no pawns of either side) or 'semi-open' (only enemy pawns). Active rooks
    are one of the clearest reasons an evaluation shifts — an open file is a highway."""
    own = set(_pawn_files(board, color))
    enemy = set(_pawn_files(board, not color))
    result: dict[str, str] = {}
    for sq in board.pieces(chess.ROOK, color):
        f = chess.square_file(sq)
        if f in own:  # a friendly pawn blocks the file — not (semi-)open for this rook
            continue
        result[chr(ord("a") + f)] = "open" if f not in enemy else "semi-open"
    return result


def _defended_by_pawn(board: chess.Board, square: int, color: chess.Color) -> bool:
    return any(
        (p := board.piece_at(a)) is not None and p.piece_type == chess.PAWN
        for a in board.attackers(color, square)
    )


def _enemy_pawn_can_challenge(
    board: chess.Board, file: int, rank: int, color: chess.Color
) -> bool:
    """Could an enemy pawn ever advance to chase a `color` piece on (file, rank)?
    True if an enemy pawn sits on an adjacent file, still ahead of the square — which
    means the square is not a true hole and a knight there is not a stable outpost."""
    for sq in board.pieces(chess.PAWN, not color):
        if abs(chess.square_file(sq) - file) != 1:
            continue
        pr = chess.square_rank(sq)
        if color == chess.WHITE and pr > rank:  # black pawn above, can come down
            return True
        if color == chess.BLACK and pr < rank:  # white pawn below, can come up
            return True
    return False


def knight_outposts(board: chess.Board, color: chess.Color) -> list[str]:
    """Squares of `color`'s knights that sit on a true outpost: advanced into enemy
    territory, defended by a friendly pawn, and on a hole no enemy pawn can attack.
    A knight like this is often worth more than a bishop — a durable, grounded plus."""
    enemy_ranks = (3, 4, 5) if color == chess.WHITE else (2, 3, 4)  # 4th–6th from the owner
    out: list[str] = []
    for sq in board.pieces(chess.KNIGHT, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if r not in enemy_ranks:
            continue
        if not _defended_by_pawn(board, sq, color):
            continue
        if _enemy_pawn_can_challenge(board, f, r, color):
            continue
        out.append(chess.square_name(sq))
    return sorted(out)


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


def _both(fn, board: chess.Board) -> dict:
    """Run a per-colour feature for White and Black: {'white': ..., 'black': ...}."""
    return {"white": fn(board, chess.WHITE), "black": fn(board, chess.BLACK)}


def positional_features(board: chess.Board) -> dict:
    """The full positional *character* of a position, for BOTH sides, as a plain
    JSON-able dict — the complete grounded snapshot the tutor reasons over for
    open-ended questions ("what are the imbalances?", "was I too passive?"). Every
    value is a verifiable fact (a count, a list of squares, a bool), so the model can
    interpret the position without us having to pre-name each concept it might raise.
    Ordered the way the engine weighs them: material/imbalance, pawns, pieces, king
    safety, activity, centre."""
    return {
        "bishop_pair": _both(has_bishop_pair, board),
        "doubled_pawns": _both(doubled_pawn_files, board),
        "isolated_pawns": _both(isolated_pawns, board),
        "backward_pawns": _both(backward_pawns, board),
        "connected_pawns": _both(connected_pawns, board),
        "passed_pawns": _both(passed_pawns, board),
        "rooks_on_open_files": _both(rooks_on_open_files, board),
        "knight_outposts": _both(knight_outposts, board),
        "trapped_pieces": _both(trapped_pieces, board),  # only the side to move is assessable
        "king_safety": _both(king_safety, board),
        "mobility": mobility(board),
        "developed_minors": _both(developed_minors, board),
        "center_control": _both(center_control, board),
    }


def positional_changes(before: chess.Board, after: chess.Board) -> list[str]:
    """Plain-English list of the positional things a move changed (before -> after).

    This is the grounded 'why' for a positional move — only true, observed
    differences, never a judgement the engine didn't make.
    """
    changes: list[str] = []
    mob_before, mob_after = mobility(before), mobility(after)
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
        new_backward = set(backward_pawns(after, color)) - set(backward_pawns(before, color))
        for sq in sorted(new_backward):
            changes.append(f"{name} now has a backward pawn on {sq}")
        new_passed = set(passed_pawns(after, color)) - set(passed_pawns(before, color))
        for sq in sorted(new_passed):
            changes.append(f"{name} now has a passed pawn on {sq}")
        ks_before = king_safety(before, color)
        ks_after = king_safety(after, color)
        if ks_after["squares_attacked_near_king"] - ks_before["squares_attacked_near_king"] >= 2:
            changes.append(f"{name}'s king is more exposed (more squares attacked near it)")
        # A lost shield pawn is a concrete, common reason a quiet-looking move hurts —
        # but only report it when the king itself stayed put, so castling and king
        # moves (which legitimately change the shield) don't read as a weakening.
        if (
            ks_before["king_square"] == ks_after["king_square"]
            and ks_after["shield_pawns"] < ks_before["shield_pawns"]
        ):
            changes.append(f"{name}'s king lost a pawn from its shield")
        if developed_minors(after, color) > developed_minors(before, color):
            changes.append(f"{name} developed a piece")
        center_delta = center_control(after, color) - center_control(before, color)
        if center_delta >= 2:
            changes.append(f"{name} gained more control of the center")
        elif center_delta <= -2:
            changes.append(f"{name} gave up control of the center")
        # Activity / coordination: a real drop in how many moves a side's pieces have
        # is a true, grounded sign the pieces got more passive. The threshold is
        # conservative (≥5 fewer AND a quarter fewer) so ordinary trades don't trip it.
        before_n, after_n = mob_before.get(name), mob_after.get(name)
        if (
            before_n is not None
            and after_n is not None
            and before_n - after_n >= 5
            and after_n <= before_n * 0.75
        ):
            changes.append(
                f"{name}'s pieces have fewer active moves ({before_n} → {after_n})"
            )
        # Rook activity: a rook that gains an (semi-)open file it didn't control before.
        rof_before, rof_after = rooks_on_open_files(before, color), rooks_on_open_files(after, color)
        for file_letter, kind in rof_after.items():
            if rof_before.get(file_letter) != kind:
                changes.append(f"{name}'s rook now controls the {kind} {file_letter}-file")
        # Knight outpost gained or lost (by net count, so a knight relocating doesn't lie).
        out_before, out_after = set(knight_outposts(before, color)), set(knight_outposts(after, color))
        if len(out_after) > len(out_before):
            sq = sorted(out_after - out_before)[0]
            changes.append(f"{name} planted a knight on a strong outpost ({sq})")
        elif len(out_after) < len(out_before):
            sq = sorted(out_before - out_after)[0]
            changes.append(f"{name}'s knight lost its outpost ({sq})")
        # A piece the move leaves trapped (assessable only for the side to move, which
        # both boards share in normal use — the mover after their move + the reply).
        if color == before.turn == after.turn:
            new_trapped = set(trapped_pieces(after, color)) - set(trapped_pieces(before, color))
            for sq in sorted(new_trapped):
                piece = after.piece_at(chess.parse_square(sq))
                pname = chess.piece_name(piece.piece_type) if piece else "piece"
                changes.append(f"{name}'s {pname} on {sq} is trapped (no safe square)")
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
