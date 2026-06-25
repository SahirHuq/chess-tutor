"""Session state — the game under discussion, with safe what-if branching.

Holds the **read-only** mainline (the real game) plus a mutable `current` board
that points at whatever position is being discussed. `goto_move` navigates the
mainline; `play_move` explores a hypothetical by branching off a *copy*, so the
real game can never be corrupted. `go_back` pops back up the branch stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import chess
import chess.pgn

import features

# PGN's seven-tag roster fills missing names with "?"; other exports use these
# stand-ins. Any of them means "we don't really know who this is."
_PLACEHOLDER_NAMES = frozenset({"", "?", "n.n.", "nn", "unknown"})
PLAYER_COLORS = ("white", "black")


def _display_name(name: str, fallback: str) -> str:
    """A human label for a PGN name, falling back when it's missing/placeholder."""
    cleaned = name.strip()
    return fallback if cleaned.lower() in _PLACEHOLDER_NAMES else cleaned


@dataclass
class SessionState:
    """All state for one review session. The mainline is never mutated.

    `player_color` records which side the *user* played ("white"/"black"), so
    player-aware tools (e.g. `find_eval_swings(scope="player")`) can scan only
    the user's moves. It is `None` until the user tells us — we never guess it
    from who won, because losing a game you played as White doesn't make you Black.
    """

    mainline_moves: list[chess.Move]  # the real game's moves, in order (read-only)
    mainline_san: list[str]  # SAN for each mainline move (for display/navigation)
    start_fen: str  # the game's starting position
    current: chess.Board  # the position currently under discussion
    current_ply: Optional[int]  # mainline ply index if on the mainline, else None
    white_player: str = "?"  # PGN [White] tag (raw; "?" when unknown)
    black_player: str = "?"  # PGN [Black] tag (raw; "?" when unknown)
    result: str = "*"  # PGN [Result] tag (raw; "*" when unfinished/unknown)
    player_color: Optional[str] = None  # which side the USER played: "white"/"black"
    branch_stack: list[tuple[chess.Board, Optional[int]]] = field(default_factory=list)
    analysis_cache: dict[str, dict] = field(default_factory=dict)

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_game(
        cls, game: chess.pgn.Game, player_color: Optional[str] = None
    ) -> "SessionState":
        """Build a session from a parsed PGN game.

        Player names + result are read from the PGN headers and preserved so the
        tutor can address the user by side. `player_color` is optional here (the
        CLI asks for it interactively after loading); pass it directly in tests.
        """
        board = game.board()
        start_fen = board.fen()
        moves: list[chess.Move] = []
        sans: list[str] = []
        for move in game.mainline_moves():
            sans.append(board.san(move))
            board.push(move)
            moves.append(move)
        session = cls(
            mainline_moves=moves,
            mainline_san=sans,
            start_fen=start_fen,
            current=chess.Board(start_fen),  # start at the opening position
            current_ply=0,
            white_player=game.headers.get("White", "?"),
            black_player=game.headers.get("Black", "?"),
            result=game.headers.get("Result", "*"),
        )
        session.set_player_color(player_color)
        return session

    # ---- player identity ----------------------------------------------------

    def set_player_color(self, color: Optional[str]) -> None:
        """Record which side the user played. `None` clears it (color unknown)."""
        if color not in (None, *PLAYER_COLORS):
            raise ValueError(f"player color must be 'white', 'black', or None — got {color!r}")
        self.player_color = color

    def opponent_color(self) -> Optional[str]:
        """The side the user did NOT play, or None when the user's color is unknown."""
        if self.player_color is None:
            return None
        return "black" if self.player_color == "white" else "white"

    def display_white(self) -> str:
        """A friendly label for the White player (falls back to 'White')."""
        return _display_name(self.white_player, "White")

    def display_black(self) -> str:
        """A friendly label for the Black player (falls back to 'Black')."""
        return _display_name(self.black_player, "Black")

    # ---- navigation ---------------------------------------------------------

    def goto_move(self, ply: int) -> dict:
        """Jump to the mainline position **after** `ply` half-moves (ply 0 = start).

        This abandons any what-if branch you were exploring (you're back on the
        real game). Returns an orientation dict, or an error dict if out of range.
        """
        if not 0 <= ply <= len(self.mainline_moves):
            return self._error(
                f"ply {ply} out of range (the game has {len(self.mainline_moves)} half-moves)"
            )
        board = chess.Board(self.start_fen)
        for move in self.mainline_moves[:ply]:
            board.push(move)
        self.current = board
        self.current_ply = ply
        self.branch_stack.clear()
        return self.orientation()

    def play_move(self, move_str: str) -> dict:
        """Play a hypothetical move from the current position, on a fresh branch.

        Validates legality via python-chess. Returns an orientation dict on
        success, or `{"ok": False, "error": "illegal: ..."}` on a bad move.
        """
        move = self.parse_move(move_str)
        if move is None:
            return self._error(
                f"illegal: '{move_str}' is not a legal move in this position "
                f"(side to move: {self._side_to_move()})"
            )
        self.branch_stack.append((self.current, self.current_ply))
        board = self.current.copy()
        board.push(move)
        self.current = board
        self.current_ply = None  # we've left the mainline
        return self.orientation()

    def go_back(self) -> dict:
        """Undo the last what-if move, returning to the previous position."""
        if not self.branch_stack:
            return self._error("already at the base position — nothing to go back to")
        self.current, self.current_ply = self.branch_stack.pop()
        return self.orientation()

    # ---- views --------------------------------------------------------------

    def orientation(self) -> dict:
        """Where are we right now? FEN + context so the model never loses track."""
        return {
            "ok": True,
            "fen": self.current.fen(),
            "side_to_move": self._side_to_move(),
            "move_number": self.current.fullmove_number,
            "last_move": self._last_move_san(),
            "on_mainline": self.current_ply is not None,
            "mainline_ply": self.current_ply,
            "next_mainline_move": self._next_mainline_move(),
            "branch_depth": len(self.branch_stack),
            "legal_move_count": self.current.legal_moves.count(),
        }

    def current_facts(self) -> dict:
        """Grounding facts (material, hanging pieces, checks, …) for `current`."""
        return features.position_facts(self.current)

    def mainline_text(self) -> str:
        """A numbered move list with ply indices, for the model's system prompt."""
        lines: list[str] = []
        for i, san in enumerate(self.mainline_san):
            move_number = i // 2 + 1
            dots = "." if i % 2 == 0 else "..."
            lines.append(f"[ply {i + 1}] {move_number}{dots}{san}")
        return "\n".join(lines) if lines else "(no moves)"

    # ---- helpers ------------------------------------------------------------

    def parse_move(self, move_str: str) -> Optional[chess.Move]:
        """Parse SAN or UCI into a move that is legal in `current` (else None)."""
        text = move_str.strip()
        for parse in (self.current.parse_san, self.current.parse_uci):
            try:
                move = parse(text)
            except (ValueError, chess.InvalidMoveError):
                continue
            if move in self.current.legal_moves:
                return move
        return None

    def _side_to_move(self) -> str:
        return "white" if self.current.turn else "black"

    def _last_move_san(self) -> Optional[str]:
        if not self.current.move_stack:
            return None
        tmp = self.current.copy()
        move = tmp.pop()
        return tmp.san(move)

    def _next_mainline_move(self) -> Optional[str]:
        if self.current_ply is None or self.current_ply >= len(self.mainline_san):
            return None
        return self.mainline_san[self.current_ply]

    @staticmethod
    def _error(message: str) -> dict:
        return {"ok": False, "error": message}
