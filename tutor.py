"""CLI chess tutor — wires the grounded engine/tools to the Gemini "voice".

Usage:
    GEMINI_API_KEY=...  .venv/bin/python tutor.py [game.pgn]
    GEMINI_MODEL=...    optional: a stronger Gemini model than the default

The model reaches the engine ONLY through the tools in `tools.py`, under a system
prompt that forbids talking about anything a tool didn't return — and every answer
is checked afterwards: a move the tools never returned sends the answer back for a
rewrite. The `GeminiTutor` adapter is the single place that knows about the
provider; swap it to change models.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
from typing import Iterable, Optional

import chess.pgn

import tools
from engine import Engine, EngineError
from session import SessionState

# Chosen from a live comparison: gemini-2.5-flash-lite skipped the probes and misread
# the board, while this model investigated (square_info, threats) and got every fact
# right. It is slower (~45s vs ~8s on an investigated answer). Set GEMINI_MODEL to
# any other model your key can use.
DEFAULT_MODEL = "gemini-3.8-flash"


def configured_model() -> str:
    """The Gemini model to use: $GEMINI_MODEL when set, else DEFAULT_MODEL."""
    return os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


# ---- the claim check: every move the answer names must come from a tool ------
#
# The prompt *asks* the model to stay grounded; this *checks* it. We pull every move
# written in the answer and compare it with the moves the tools returned (plus the
# game's own moves and the user's question). Plain pawn pushes like "e4" are skipped
# on purpose: in prose they are indistinguishable from square names ("the pawn on
# e4"), and a false alarm would make the tutor rewrite correct answers.
_MOVE_MENTION = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(O-O-O|O-O|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|[a-h]x[a-h][1-8](?:=[QRBN])?|[a-h][18]=[QRBN])"
    r"[+#]?(?![A-Za-z0-9])"
)


def _move_key(san: str) -> str:
    """Reduce a move to piece + destination (+ promotion) so harmless spelling
    differences — "Nbd2" vs "Nd2", "Nxe5" vs "Ne5", a missing "+" — still match."""
    if san.startswith("O-O"):
        return san
    dest = re.findall(r"[a-h][1-8]", san)[-1]
    promo = san.split("=")[1] if "=" in san else ""
    # The piece letter — or, for a pawn capture/promotion, the file it came from.
    return f"{san[0]}{dest}{promo}"


def move_mentions(text: str) -> dict[str, str]:
    """Moves written in `text`, as {key: the move as written} (first spelling wins)."""
    found: dict[str, str] = {}
    for match in _MOVE_MENTION.finditer(text):
        found.setdefault(_move_key(match.group(1)), match.group(1))
    return found


def _strings_in(value: object) -> Iterable[str]:
    """Every string inside a tool result (keys and values, at any depth). Scanning
    the raw strings — not a JSON dump — keeps escapes like "\\n" from gluing a letter
    onto a move and hiding it from the move pattern."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings_in(key)
            yield from _strings_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings_in(item)


def unverified_moves(answer: str, grounded: Iterable[str]) -> list[str]:
    """Moves the answer names that no tool, game move, or user question supplied."""
    known = set(grounded)
    return [written for key, written in move_mentions(answer).items() if key not in known]

SYSTEM_TEMPLATE = """\
You are a friendly chess tutor reviewing THIS game with a player who is still \
learning. Explain warmly, and be THOROUGH: make the player understand exactly \
what went wrong in their thinking by walking through the concrete line and the \
better plan, not vague adjectives. A few short paragraphs are good when the tools \
gave you real detail to teach; don't pad, but don't be vaguely brief either.

{player_context}

You cannot picture a chessboard in your head, so you must NEVER state a move, \
evaluation, or line from memory. Only repeat facts a TOOL returns to you this \
turn. If you haven't called a tool yet, you don't know anything about the \
position — call one first.

Choosing a tool:
- When the user asks why a move was good, bad, a blunder, or a mistake (e.g. \
"why is 9.Nxd4 a blunder?"), immediately call explain_move("9.Nxd4") — pass the \
exact move as the user wrote it, including the move number when present. ALWAYS call it — even if you discussed \
that move earlier in this conversation; NEVER answer a question about a move's \
quality from memory or a previous turn (the verdict and the line are too easy to \
misremember). Do NOT ask the user to describe the position; you \
already have the whole game. This is your main tool: it returns the better \
options that were available AND what the move actually led to, so you narrate \
the difference.
- When the user asks which of two moves is better, or why one move is better \
than another (e.g. "is Nf3 or Nc3 better here?", "why is Bd3 better than Be2?"): \
first call goto_move(ply) to the position the choice is from (it must be that \
player's turn), then call compare_moves("Nf3", "Nc3"). It scores BOTH from the \
same position and returns, for each, a verdict, the centipawns it loses vs the \
engine's best, what it does, its consequence, and its position_changes — plus \
comparison.better_move / centipawn_gap and the engine's own engine_best pick. \
Lead with which move is better and the gap, then explain WHY from each move's \
data, exactly as for a single move. If a move is illegal there, relay the reason \
and do NOT invent a line.
- When the user asks a VAGUE question that does NOT name a move — "where did I go \
wrong?", "what was my biggest mistake?", "why did my eval drop?", "what was the \
turning point?" — call find_eval_swings() instead of guessing a move. Choose its \
`scope` by WHOSE moves the question is about: \
  · "where did I go wrong?", "my biggest mistake", "did I blunder?" → \
scope="player" (the user's own moves only — the default). \
  · "what did my opponent miss?", "where did THEY go wrong?" → scope="opponent". \
  · neutral questions about the game itself, e.g. "what was the turning point?" → \
scope="both". \
For "around move N" / "what happened near move 18?", also pass around_move=N to \
focus the scan. It returns `swings`: the worst moves, biggest drop first, EACH \
with the same fields as explain_move (plus `side`). Lead with the biggest swing, \
then explain it with the same VERDICT → WHY → PRINCIPLE structure you use for a \
single move (its recommended_plan, position_changes, consequence). Mention the \
next worst only if it helps. Respect `note`: if it says the game was cleanly \
played, say so honestly rather than inventing a mistake. If the result is \
ok=false because your colour is unknown, tell the user and ask which side they \
played — do NOT scan both sides and pass it off as theirs.
- For "what if I had played X instead?": call goto_move(ply) to the position \
before the move (using the ply numbers below), then play_move("X"), then \
analyze(). If play_move says illegal, tell the user why and do NOT invent a \
line. Call go_back() to undo.
- To simply evaluate wherever the board currently is (a quick "who's better / what's \
the best move" check), call analyze().
- When the user asks an OPEN-ENDED or CONCEPTUAL question about a position that does \
NOT name a move and is NOT about finding a mistake — "is my position better here?", \
"what are the imbalances?", "was I too passive?", "is my bishop good or bad?", "what's \
my plan?", "what are the weaknesses here?" — first goto_move(ply) to the position they \
mean (if they named one), then call explain_position(). It returns the engine's verdict \
and best plan PLUS the full positional character for BOTH sides (bishop pair, pawn \
weaknesses, open files, knight outposts, trapped pieces, king safety, activity, centre). \
REASON over ALL of it to teach — connect the facts into a plan or an explanation, even \
for ideas no single field names — but keep every concrete claim (a move, an eval, a \
square, a piece) grounded in what it returned; never invent one.
- INVESTIGATE like a coach when the named fields don't explain a move — e.g. \
explain_move's `tactics` and `position_changes` are empty but it lost material or a \
lot of evaluation, or the user asks "but WHY?". Form a hypothesis about the idea, then \
CHECK it with the probes before you say it. explain_move leaves the board on the \
position just BEFORE the move, so you can probe there directly:
  · try_line("Nxe5 Qxe5 Rxd8+") — walks a line from the current board (without \
moving it) and reports after EACH move what it captures and attacks, whether it \
checks, and what is left hanging. Walk the played move followed by the moves in \
`consequence` to find the exact moment material falls.
  · square_info("e5") — who attacks and defends a square (each marked \
`pinned_to_king`), what the piece there attacks and `defends`, and whether it is \
attacked by a cheaper piece. Use it to test an overloaded defender (one piece whose \
`defends` covers two attacked things), a pinned defender that can't really recapture, \
or a piece attacked by a cheaper piece.
  · threats() — what the opponent would play if it were their move now. Use it for \
"what did I miss?" or when a quiet move lost because it ignored a threat; if its \
`serious` is false, say there was no real threat.
  When the probe results show it, NAME the idea — overloaded defender, removing the \
defender, discovered attack, skewer, back-rank weakness, piece attacked by a cheaper \
piece, ignored threat — and walk through the moves that prove it. Name an idea ONLY \
when the probes confirm it, and mention only moves, squares and pieces a tool \
returned. Probes never change the verdict: keep explain_move's `verdict` and \
`centipawns_lost`.

explain_move works for ANY move and gives you everything to TEACH, not just \
label. Your job is to help the player understand and improve. Structure each \
answer:
1. VERDICT first. Use the tool's `verdict` field EXACTLY — best move / good \
move / inaccuracy / mistake / blunder. NEVER soften or upgrade it: if the tool \
says "blunder", it is a blunder even when material ends up even — a move can lose \
almost no material yet still be a blunder because the EVALUATION collapses. Always \
tell the player how much it cost from `centipawns_lost`, in pawns (e.g. 307 \
centipawns ≈ "about 3 pawns of evaluation"; 0 = the engine's top move).
2. WHY — grounded ONLY in the tool data:
   - Describe the better move's idea move-by-move from `recommended_plan.plan`. \
Each entry lists EXACTLY what that move `captures` and `attacks`, plus `side` \
(who plays it) and `from` (the square the piece came from) — say only what is \
listed there. Attribute every move to its `side`, and name an origin square \
only from `from` (e.g. plan = [white h3 attacks bishop on g4] [black Bh5] \
[white g4 attacks bishop on h5] → "h3 hits your bishop so it retreats to h5, \
then g4 chases it again, gaining space"). Also use `recommended_plan.achieves` \
for positional gains.
   - Then contrast with the played move: what it did or gave up, from \
`position_changes`, `consequence`, and `actual_move_does`.
   - NAME any concrete tactic in `tactics` — a fork, a pin, or a piece left \
hanging. These are the sharp reason a move wins or loses material, so state them \
plainly ("this hangs your bishop on d1", "Nc6 forks the queen and the rook", "Re1 \
pins the knight to the king"). Only list tactics that appear in `tactics`. \
   - For a SMALL drop, do NOT stop at "slightly worse": if `position_changes` \
names a subtle reason (gave up the center, a rook seized an open file, a knight \
reached an outpost, a piece got trapped, pieces have fewer active moves, a worse \
pawn — doubled / isolated / backward — the king a touch looser), give THAT and \
say why it matters — this is the kind of quiet detail that teaches the most. \
   - For a good/best move, explain its own advantages the same way.

`attacks` means a piece is hit and may have to move — say "wins"/"traps" ONLY \
if `consequence`'s material result confirms a gain; otherwise say "attacks", \
"hits", "forces it back", or "gains a tempo".
3. THE PRINCIPLE — when the data names a concept (bishop pair, development, \
control of the center, king safety, doubled/isolated/passed pawns, winning \
material), state it AND briefly why it matters in general, so the player learns \
something reusable (e.g. "keeping both bishops gives long-term chances on both \
colors"). Prefer the SPECIFIC concept the tools named (e.g. "you gave up the \
bishop pair") over generic advice (e.g. "be careful when trading"). This is the \
part that improves their thinking.

Material: respect `consequence`'s NET result — if material stays even it is a \
trade or a positional point, never a material "win". If a line's `ends_mid_exchange` \
is true, the net is NOT final: don't quote it as a win — walk the exchange with \
try_line instead. Evaluations are White's point of view: positive favors White, \
negative favors Black.

HONESTY OVER FABRICATION: use only facts the tools returned. Never invent \
positional reasons (no guessing about "activity", "weak squares", "a more active \
queen"), never state a piece's square, a recapture, or a positional claim the \
tools did not report, and never say a move attacks or defends a piece that is not \
in that move's `attacks`/`captures` list or a probe's `attacks`/`defends`/\
`attacked_by`/`defended_by` result — if you catch yourself guessing, stop, and probe \
instead. \
When the grounded reasons are thin (empty `position_changes` and `achieves`, even \
material), let `centipawns_lost` decide HOW you describe it — do NOT let thin \
reasons push you into softening the verdict: \
  · a small loss (best / good move) → if `tactics` or `position_changes` names \
something concrete, give THAT as the subtle reason (a slightly worse pawn, the \
center loosened, less active pieces) and why it matters; only when BOTH are empty \
and material is even, say plainly it is fine and the difference is subtle. \
  · a large loss (inaccuracy / mistake / blunder) → KEEP that verdict; name the \
tactic from `tactics` if present (the fork/pin/hung piece), and show the exact \
continuation from `consequence` (the real moves and what they capture or threaten) \
as the concrete reason. If material stays even, say the cost is in what the \
position becomes (still `centipawns_lost` of evaluation), not in lost material.

Moves in the game (use these ply numbers with the tools; ply N = the Nth \
half-move):
{move_list}
"""


class GeminiTutor:
    """Adapter around google-genai with a *manual* function-calling loop.

    We drive the tool loop ourselves (rather than the SDK's automatic mode) so
    every engine call is logged via `tools`, tool errors are handled gracefully
    instead of crashing, transient 5xx errors are retried, and we can nudge the
    model when it returns an empty answer after a tool call. This is the ONLY
    module that knows the provider — reimplement it to swap models.

    After the model answers, `ask` checks every move it names against the moves the
    tools have returned in this conversation (plus the game's moves and the user's
    own words). An unverified move sends the answer back ONCE for a rewrite; if it
    survives that, the answer carries a visible caution instead of passing silently.
    `client` is injectable so this loop can be tested without the network.
    """

    # Investigating (explain_move, then try_line / square_info / threats) takes more
    # rounds than the old fixed-verdict flow; the cap still stops a looping model.
    MAX_TOOL_ROUNDS = 12

    def __init__(
        self,
        system_instruction: str,
        tool_fns: list,
        api_key: str,
        known_moves: Iterable[str] = (),
        model: Optional[str] = None,
        client=None,
    ) -> None:
        from google.genai import errors, types

        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self._types = types
        self._errors = errors
        self._client = client
        self._model = model or configured_model()
        self._tools = {fn.__name__: fn for fn in tool_fns}
        self._config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tool_fns,  # used for the schemas only; we execute the calls
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        self._contents: list = []  # full conversation, grows across turns
        # Move keys the answer may name: the game's own moves, then everything the
        # tools return and the user writes, accumulated across the conversation.
        self._grounded: set[str] = set()
        for san in known_moves:
            self._grounded.update(move_mentions(san))

    def ask(self, question: str) -> str:
        self._grounded.update(move_mentions(question))
        self._append_user(question)
        answer = self._answer_with_tools()
        unverified = unverified_moves(answer, self._grounded)
        if not unverified:
            return answer
        self._append_user(
            "Check your answer: it mentions " + ", ".join(unverified) + ", which no tool "
            "returned in this conversation. Either call a tool that confirms them, or "
            "rewrite the answer without them. Only state facts the tools returned."
        )
        answer = self._answer_with_tools()
        still_unverified = unverified_moves(answer, self._grounded)
        if still_unverified:
            answer += (
                "\n\n(Caution: " + ", ".join(still_unverified) + " did not come from the "
                "engine tools — treat those moves with suspicion.)"
            )
        return answer

    def _append_user(self, text: str) -> None:
        types = self._types
        self._contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))

    def _answer_with_tools(self) -> str:
        """Let the model call tools until it produces a text answer (or hits the cap)."""
        types = self._types
        for _ in range(self.MAX_TOOL_ROUNDS):
            response = self._generate()
            content = response.candidates[0].content
            self._contents.append(content)
            calls = [p.function_call for p in (content.parts or []) if p.function_call]
            if not calls:
                text = (response.text or "").strip()
                return text or self._nudge_for_answer()
            self._contents.append(
                types.Content(role="user", parts=self._run_calls(calls))
            )
        return "(stopped after too many tool calls — try rephrasing your question.)"

    def _generate(self):
        last_error = None
        for attempt in range(4):
            try:
                return self._client.models.generate_content(
                    model=self._model, contents=self._contents, config=self._config
                )
            except self._errors.ClientError as exc:  # 4xx — don't blind-retry
                if getattr(exc, "code", None) == 429:
                    raise RuntimeError(
                        "Gemini's free-tier rate limit was hit (it allows limited "
                        "requests per minute and per day). Wait a minute and try "
                        "again, or use a billed API key for heavier use."
                    ) from None
                raise
            except self._errors.ServerError as exc:  # transient 5xx / overloaded
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        raise last_error

    def _run_calls(self, calls) -> list:
        types = self._types
        parts = []
        for call in calls:
            fn = self._tools.get(call.name)
            args = dict(call.args or {})
            if fn is None:
                result = {"ok": False, "error": f"unknown tool: {call.name}"}
            else:
                try:
                    result = fn(**args)
                except Exception as exc:  # report tool errors to the model, don't crash
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            # Whatever a tool returned is grounded — the answer may name these moves.
            for text in _strings_in(result):
                self._grounded.update(move_mentions(text))
            parts.append(
                types.Part.from_function_response(name=call.name, response={"result": result})
            )
        return parts

    def _nudge_for_answer(self) -> str:
        """Some small models go silent after a tool call — ask once for the answer."""
        types = self._types
        self._contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(
                    text="Now answer my question in a few clear sentences, "
                    "using the tool results above."
                )],
            )
        )
        response = self._generate()
        self._contents.append(response.candidates[0].content)
        return (response.text or "").strip() or "(no answer)"


def session_from_pgn_text(text: str) -> SessionState:
    """Parse PGN text (headers, comments, clocks, NAGs, variations all OK)."""
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None or not list(game.mainline_moves()):
        raise ValueError("no playable moves found — is this a complete PGN?")
    return SessionState.from_game(game)


def load_session(pgn_path: str) -> SessionState:
    with open(pgn_path, "r", encoding="utf-8") as handle:
        return session_from_pgn_text(handle.read())


def read_pasted_pgn() -> str:
    """Read a multi-line PGN paste until a lone 'END' line or EOF (Ctrl-D)."""
    print(
        "\nPaste your game's PGN below (a Lichess or Chess.com export works).\n"
        "When you're done, type END on its own line (or press Ctrl-D):\n"
    )
    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
    except EOFError:
        print()
    return "\n".join(lines)


def interactive_load() -> SessionState:
    """Prompt for a PGN: a file path, a pasted game, or the bundled demo."""
    while True:
        choice = input(
            "Load a game — type 'paste' to paste a PGN, give a .pgn file path, "
            "or 'sample' for the demo: "
        ).strip()
        try:
            if choice.lower() == "sample":
                return load_session("sample.pgn")
            if choice.lower() == "paste":
                return session_from_pgn_text(read_pasted_pgn())
            if choice:
                return load_session(choice)
        except (ValueError, OSError) as exc:
            print(f"  Couldn't load that: {exc}\n  Try again.\n")


def compact_moves(session: SessionState) -> str:
    """The game in normal notation: '1. e4 e5 2. Nf3 Nc6 ...'."""
    parts: list[str] = []
    for i, san in enumerate(session.mainline_san):
        parts.append(f"{i // 2 + 1}. {san}" if i % 2 == 0 else san)
    return " ".join(parts)


def build_player_context(session: SessionState) -> str:
    """Tell the model who's who and which side is the user — so it says 'you' for
    the user's moves and 'your opponent' for the other side, and never confuses the
    two. When the user's colour is unknown, it must ask rather than guess."""
    white, black = session.display_white(), session.display_black()
    if session.player_color is None:
        return (
            f"This game is {white} (White) vs {black} (Black); result {session.result}. "
            "You do NOT yet know which side the user played — never infer it from who "
            'won. If they ask about "my" moves or mistakes, ask whether they were '
            "White or Black before scanning their moves."
        )
    you = session.player_color
    opp = session.opponent_color()
    you_name = white if you == "white" else black
    opp_name = black if you == "white" else white
    return (
        f"The user is {you_name}, who played the {you} pieces; their opponent is "
        f"{opp_name}, who played {opp}. The game result was {session.result}. Use "
        f'"you"/"your" ONLY for {you}\'s moves and "your opponent" for {opp}\'s moves '
        "— never describe an opponent's move as the user's."
    )


def build_system_prompt(session: SessionState) -> str:
    return SYSTEM_TEMPLATE.format(
        player_context=build_player_context(session),
        move_list=session.mainline_text(),
    )


def prompt_player_color(session: SessionState, ask=input, show=print) -> Optional[str]:
    """Show both players and ask which side the user played. Returns "white",
    "black", or None (skipped / no answer). We ask rather than guess from the
    result — losing as White doesn't make you Black."""
    show(
        f"\nThis game: {session.display_white()} (White) vs "
        f"{session.display_black()} (Black) — result {session.result}."
    )
    while True:
        try:
            choice = ask("Which side did you play? [white/black/skip]: ").strip().lower()
        except EOFError:
            return None
        if choice in {"w", "white"}:
            return "white"
        if choice in {"b", "black"}:
            return "black"
        if choice in {"s", "skip", ""}:
            return None
        show("  Please type 'white', 'black', or 'skip'.")


def _terminal_input():
    """An `input`-like callable bound to the controlling terminal.

    For `pbpaste | tutor.py -` the PGN consumed stdin, so a plain `input()` would
    hit EOF; we reach the user's keyboard through /dev/tty instead. Returns None
    when there is no terminal to ask on (so the caller skips the prompt)."""
    if sys.stdin.isatty():
        return input
    try:
        tty = open("/dev/tty", "r")
    except OSError:
        return None

    def ask(prompt: str = "") -> str:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        line = tty.readline()
        if not line:
            raise EOFError
        return line.rstrip("\n")

    return ask


def resolve_player_color(session: SessionState, color_flag: Optional[str]) -> None:
    """Settle which side the user played: an explicit --color wins; otherwise ask
    on the terminal. If neither is possible (piped in, no tty, no flag), leave it
    unknown — player-specific scans will then return a clear error."""
    if color_flag:
        session.set_player_color(color_flag)
        return
    ask = _terminal_input()
    if ask is None:
        print(
            "[tutor] no terminal to ask which side you played — pass --color "
            "white|black to enable 'where did I go wrong?' questions.",
            file=sys.stderr,
        )
        return
    session.set_player_color(prompt_player_color(session, ask=ask))


def repl(tutor: GeminiTutor, session: SessionState) -> None:
    print(f"\nLoaded your game ({len(session.mainline_san)} half-moves):")
    print(f"{session.display_white()} (White) vs {session.display_black()} (Black) — {session.result}")
    if session.player_color:
        print(f"You played {session.player_color}.")
    print(compact_moves(session))
    print(
        "\nAsk me anything about it — e.g. \"was 12.Nf3 a good move?\", "
        '"why did my eval drop on move 18?", or "what if I had castled instead?". '
        'Type "quit" to exit.\n'
    )
    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            return
        try:
            answer = tutor.ask(question)
        except Exception as exc:  # keep the REPL alive on transient API errors
            print(f"\n[error talking to the model: {exc}]\n", file=sys.stderr)
            continue
        print(f"\ntutor > {answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Conversational AI chess tutor")
    parser.add_argument(
        "pgn",
        nargs="?",
        default=None,
        help="path to a .pgn file, or '-' to read PGN from stdin "
        "(omit to choose interactively: paste / file / sample)",
    )
    parser.add_argument(
        "--color",
        choices=["white", "black"],
        default=None,
        help="which side YOU played (skips the prompt; required to enable "
        "'my mistakes' questions when piping a PGN via stdin)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "[tutor] GEMINI_API_KEY is not set. Get a free key at "
            "aistudio.google.com, then run:  export GEMINI_API_KEY=..."
        )

    try:
        if args.pgn == "-":
            session = session_from_pgn_text(sys.stdin.read())
        elif args.pgn:
            session = load_session(args.pgn)
        else:
            session = interactive_load()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"[tutor] couldn't load that game: {exc}")

    resolve_player_color(session, args.color)

    try:
        with Engine() as engine:
            tools.bind(session, engine)
            tutor = GeminiTutor(
                build_system_prompt(session), tools.ALL_TOOLS, api_key,
                known_moves=session.mainline_san,
            )
            repl(tutor, session)
    except EngineError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
