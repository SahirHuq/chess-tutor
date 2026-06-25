"""Stockfish wrapper — the deterministic *judgement* half of the brain.

Thin layer over `python-chess`'s UCI engine interface. Turns a board position
into a structured, JSON-able evaluation (top candidate moves + scores + lines).
It never decides legality or tracks geometry — that's `python-chess`/`features.py`.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Optional

import chess
import chess.engine

DEFAULT_DEPTH = 18
DEFAULT_MULTIPV = 3
PV_PLIES = 8  # how many half-moves of the principal variation to show


class EngineError(RuntimeError):
    """Raised when the engine can't be found or used."""


def find_stockfish() -> str:
    """Locate the Stockfish binary via $STOCKFISH_PATH or PATH."""
    path = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
    if not path:
        raise EngineError(
            "[find_stockfish] Stockfish not found. Install it "
            "(`brew install stockfish`) or set STOCKFISH_PATH."
        )
    return path


def _eval_str(white_cp: Optional[int], mate_in: Optional[int]) -> str:
    """A correctly-signed, human-readable evaluation from White's POV."""
    if mate_in is not None:
        side = "white" if mate_in > 0 else "black"
        return f"mate in {abs(mate_in)} for {side}"
    assert white_cp is not None
    pawns = white_cp / 100
    if white_cp > 30:
        who = "white better"
    elif white_cp < -30:
        who = "black better"
    else:
        who = "roughly equal"
    return f"{pawns:+.2f} ({who})"


@dataclass
class Line:
    """One candidate move and the line that follows it, judged by the engine."""

    rank: int  # 1 = engine's top choice
    move: str  # the candidate move in SAN (e.g. "Nf3")
    pv: str  # the principal variation in SAN (e.g. "Nf3 Nc6 Bb5 ...")
    white_cp: Optional[int]  # centipawns from White's POV (None if forced mate)
    mate_in: Optional[int]  # mate-in-N from White's POV (+white / -black; None otherwise)
    moves: list = field(default_factory=list)  # raw Move objects (for annotation; not serialized)

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "move": self.move,
            "eval": _eval_str(self.white_cp, self.mate_in),
            "line": self.pv,
            "white_centipawns": self.white_cp,
            "mate_in": self.mate_in,
        }


@dataclass
class Analysis:
    """The engine's verdict on a position: a ranked list of candidate moves."""

    fen: str
    side_to_move: str  # "white" or "black"
    depth: int
    lines: list[Line]
    outcome: Optional[str] = None  # set for game-over positions, else None

    def to_dict(self) -> dict:
        return {
            "fen": self.fen,
            "side_to_move": self.side_to_move,
            "depth": self.depth,
            "outcome": self.outcome,
            "best_moves": [line.to_dict() for line in self.lines],
        }


def _describe_outcome(board: chess.Board) -> Optional[str]:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.termination == chess.Termination.CHECKMATE:
        winner = "white" if outcome.winner else "black"
        return f"checkmate — {winner} wins"
    return outcome.termination.name.lower().replace("_", " ")


class Engine:
    """Owns one long-lived Stockfish process. Use as a context manager:

        with Engine() as eng:
            analysis = eng.analyse(board)
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or find_stockfish()
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def __enter__(self) -> "Engine":
        self._engine = chess.engine.SimpleEngine.popen_uci(self._path)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def analyse(
        self,
        board: chess.Board,
        multipv: int = DEFAULT_MULTIPV,
        depth: int = DEFAULT_DEPTH,
    ) -> Analysis:
        if self._engine is None:
            raise EngineError(
                "[Engine.analyse] engine not started — use `with Engine() as eng:`"
            )

        side = "white" if board.turn else "black"
        if board.is_game_over(claim_draw=True):
            return Analysis(board.fen(), side, depth, [], _describe_outcome(board))

        infos = self._engine.analyse(
            board, chess.engine.Limit(depth=depth), multipv=multipv
        )

        lines: list[Line] = []
        for i, info in enumerate(infos):
            score = info["score"].white()
            moves = info.get("pv", [])
            if score.is_mate():
                white_cp, mate_in = None, score.mate()
            else:
                white_cp, mate_in = score.score(), None
            move_san = board.san(moves[0]) if moves else ""
            pv_san = board.variation_san(moves[:PV_PLIES]) if moves else ""
            lines.append(
                Line(i + 1, move_san, pv_san, white_cp, mate_in, moves=list(moves[:PV_PLIES]))
            )

        return Analysis(board.fen(), side, depth, lines)
