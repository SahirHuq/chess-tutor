"""Local web frontend — a thin *bridge* from the browser to the existing brain.

Chess is visual, but the tutor lived in the terminal. This module adds a small,
single-user, localhost web UI **without** rewriting any chess logic: it reuses the
exact same `SessionState`, `Engine`, `tools.bind`, and `GeminiTutor` the CLI uses.
Like `tools.py`, it is a bridge, never a second source of truth — it adds no chess
reasoning of its own, so the anti-hallucination contract is untouched.

`ReviewApp` holds all the logic and returns plain JSON-able dicts (no HTTP types),
so it is unit-testable without a server; the request handler below is a thin shell.
Every public method runs under one lock: the model navigates the session *during*
an `ask`, so a concurrent `state` read must not see a half-moved board.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import chess
import chess.svg

import tools
import tutor
from engine import Engine, EngineError
from session import SessionState

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
SAMPLE_PGN = os.path.join(BASE_DIR, "sample.pgn")

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}


class ReviewApp:
    """All the web app's state + behaviour, independent of HTTP.

    `engine`/`api_key` may be None: the board, PGN loading, colour, and move
    navigation never need them, so the UI still works with neither installed —
    only `ask` (the model) requires both, and it returns a clear error if missing.
    """

    def __init__(self, engine: Optional[Engine] = None, api_key: Optional[str] = None) -> None:
        self.engine = engine
        self.api_key = api_key
        self.session: Optional[SessionState] = None
        self.view_ply = 0  # the ply the USER is looking at (the model may roam during ask)
        self._tutor: Optional[tutor.GeminiTutor] = None
        self._lock = threading.RLock()  # re-entrant: load_game etc. call _state_locked

    # ---- actions (each returns a JSON-able dict; never raises to the caller) --

    def load_game(self, pgn: Optional[str] = None, sample: bool = False) -> dict:
        with self._lock:
            try:
                if sample:
                    session = tutor.load_session(SAMPLE_PGN)
                else:
                    session = tutor.session_from_pgn_text(pgn or "")
            except (ValueError, OSError) as exc:
                return {"ok": False, "error": f"Couldn't load that PGN: {exc}"}
            self.session = session
            self.view_ply = 0
            self._tutor = None  # fresh game → fresh conversation
            if self.engine is not None:
                tools.bind(self.session, self.engine)
            return self._state_locked()

    def set_color(self, color: Optional[str]) -> dict:
        with self._lock:
            if self.session is None:
                return {"ok": False, "error": "No game loaded yet."}
            try:
                self.session.set_player_color(color)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            self._tutor = None  # the system prompt depends on which side the user played
            return self._state_locked()

    def goto(self, ply: object) -> dict:
        with self._lock:
            if self.session is None:
                return {"ok": False, "error": "No game loaded yet."}
            try:
                ply_int = int(ply)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return {"ok": False, "error": "ply must be an integer."}
            result = self.session.goto_move(ply_int)
            if not result.get("ok", True):
                return result  # out-of-range error, surfaced verbatim
            self.view_ply = ply_int
            return self._state_locked()

    def ask(self, question: Optional[str]) -> dict:
        with self._lock:
            if self.session is None:
                return {"ok": False, "error": "No game loaded yet."}
            text = (question or "").strip()
            if not text:
                return {"ok": False, "error": "Ask a question first."}
            if not self.api_key:
                return {"ok": False, "error": (
                    "GEMINI_API_KEY is not set, so the tutor can't answer. Get a free "
                    "key at aistudio.google.com, export it, and restart the web app."
                )}
            if self.engine is None:
                return {"ok": False, "error": (
                    "Stockfish isn't available, so I can't analyse positions. Install it "
                    "(brew install stockfish) or set STOCKFISH_PATH, then restart."
                )}
            tools.bind(self.session, self.engine)
            if self._tutor is None:
                try:
                    self._tutor = tutor.GeminiTutor(
                        tutor.build_system_prompt(self.session), tools.ALL_TOOLS, self.api_key
                    )
                except Exception as exc:  # bad key / import / client init — report, don't crash
                    return {"ok": False, "error": f"Couldn't start the tutor: {exc}"}
            try:
                answer = self._tutor.ask(text)
            except Exception as exc:  # transient API / model error — keep the server alive
                return {"ok": False, "error": f"Error talking to the model: {exc}"}
            finally:
                # The model navigated the board while analysing; put it back where the
                # user was looking so the next /api/state render doesn't jump.
                self.session.goto_move(self.view_ply)
            return {"ok": True, "answer": answer}

    def state(self) -> dict:
        with self._lock:
            return self._state_locked()

    # ---- views ---------------------------------------------------------------

    def _state_locked(self) -> dict:
        """The full UI snapshot. Must be called while holding `self._lock`."""
        if self.session is None:
            return {
                "ok": True, "loaded": False,
                "engine_available": self.engine is not None,
                "api_key_present": bool(self.api_key),
            }
        s = self.session
        s.goto_move(self.view_ply)  # guarantee the rendered board matches the user's view
        board = s.current
        moves = [
            {
                "ply": i + 1,
                "san": san,
                "move_number": (i + 1 + 1) // 2,
                "side": "white" if (i + 1) % 2 == 1 else "black",
            }
            for i, san in enumerate(s.mainline_san)
        ]
        svg, svg_error = self._render_svg(board)
        return {
            "ok": True,
            "loaded": True,
            "white": s.display_white(),
            "black": s.display_black(),
            "result": s.result,
            "player_color": s.player_color,
            "moves": moves,
            "current_ply": self.view_ply,
            "side_to_move": "white" if board.turn else "black",
            "move_number": board.fullmove_number,
            "last_move_san": s._last_move_san(),
            "board_svg": svg,
            "svg_error": svg_error,
            "engine_available": self.engine is not None,
            "api_key_present": bool(self.api_key),
        }

    def _render_svg(self, board: chess.Board) -> tuple[Optional[str], Optional[str]]:
        """Render the board to an SVG string, oriented to the user's side. Any render
        failure becomes a clear error string instead of crashing the request."""
        try:
            lastmove = board.peek() if board.move_stack else None
            check = board.king(board.turn) if board.is_check() else None
            orientation = (
                chess.BLACK
                if (self.session is not None and self.session.player_color == "black")
                else chess.WHITE
            )
            svg = chess.svg.board(
                board, size=400, lastmove=lastmove, check=check, orientation=orientation
            )
            return svg, None
        except Exception as exc:
            return None, f"Couldn't render the board: {exc}"


class _Handler(BaseHTTPRequestHandler):
    """Thin HTTP shell: parse the request, dispatch to `self.server.app`, send JSON.

    All chess logic lives in `ReviewApp`; this class only translates HTTP <-> dicts.
    """

    server_version = "ChessTutorWeb"

    @property
    def _app(self) -> ReviewApp:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, *_args: object) -> None:
        """Silence per-request logging so the dimmed `[engine tool]` grounding logs
        (printed by the tools) stay readable in the console."""

    # ---- routing -------------------------------------------------------------

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_file("index.html")
        elif self.path.startswith("/static/"):
            self._send_file(self.path[len("/static/"):])
        elif self.path == "/api/state":
            self._send_json(self._app.state())
        else:
            self._send_json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self) -> None:
        body = self._read_json()
        if body is None:
            self._send_json({"ok": False, "error": "invalid JSON body"}, status=400)
            return
        if self.path == "/api/load-game":
            result = self._app.load_game(pgn=body.get("pgn"), sample=bool(body.get("sample")))
        elif self.path == "/api/set-color":
            result = self._app.set_color(body.get("color"))
        elif self.path == "/api/goto":
            result = self._app.goto(body.get("ply"))
        elif self.path == "/api/ask":
            result = self._app.ask(body.get("question"))
        else:
            self._send_json({"ok": False, "error": "not found"}, status=404)
            return
        self._send_json(result)

    # ---- helpers -------------------------------------------------------------

    def _read_json(self) -> Optional[dict]:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            return None
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _send_json(self, obj: dict, status: int = 200) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, name: str) -> None:
        # Sanitise: only plain filenames inside WEB_DIR, never a traversal.
        safe = os.path.normpath(name).lstrip(os.sep)
        if safe.startswith("..") or os.sep in safe:
            self._send_json({"ok": False, "error": "not found"}, status=404)
            return
        path = os.path.join(WEB_DIR, safe)
        if not os.path.isfile(path):
            self._send_json({"ok": False, "error": "not found"}, status=404)
            return
        ctype = _CONTENT_TYPES.get(os.path.splitext(safe)[1], "application/octet-stream")
        with open(path, "rb") as handle:
            payload = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _start_engine() -> Optional[Engine]:
    """Open a long-lived Stockfish process, or None (with a clear note) if absent.
    Navigation/loading still work without it; only `ask` needs analysis."""
    try:
        engine = Engine()
        engine.__enter__()
        return engine
    except EngineError as exc:
        print(f"[web] {exc}")
        print("[web] Analysis is disabled until Stockfish is available; "
              "loading and navigation still work.")
        return None


def main() -> None:
    host = os.environ.get("CHESS_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("CHESS_WEB_PORT", "8000"))
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[web] GEMINI_API_KEY is not set — the board and move list work, but the "
              "tutor chat will return a clear error until you export a key and restart.")

    engine = _start_engine()
    app = ReviewApp(engine=engine, api_key=api_key)
    server = ThreadingHTTPServer((host, port), _Handler)
    server.app = app  # type: ignore[attr-defined]
    print(f"Chess tutor web app running at http://{host}:{port}")
    print("Open it in your browser; press Ctrl-C here to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] shutting down.")
    finally:
        server.server_close()
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    main()
