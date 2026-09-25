"""Tests for the tutor prompt and adapter-adjacent behavior."""

import io
from types import SimpleNamespace

import chess.pgn
from google.genai import types

import tools
import tutor
from session import SessionState
from tutor import build_system_prompt, prompt_player_color


def _session(pgn: str, player_color=None) -> SessionState:
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    return SessionState.from_game(game, player_color=player_color)


def test_system_prompt_preserves_move_number_in_explain_move_example():
    game = chess.pgn.read_game(io.StringIO("1. e4 e5 1/2-1/2"))
    assert game is not None
    prompt = build_system_prompt(SessionState.from_game(game))

    assert 'explain_move("9.Nxd4")' in prompt
    assert "including the move number when present" in prompt


def test_system_prompt_routes_vague_mistake_questions_to_eval_swings():
    game = chess.pgn.read_game(io.StringIO("1. e4 e5 1/2-1/2"))
    assert game is not None
    prompt = build_system_prompt(SessionState.from_game(game))

    assert "call find_eval_swings()" in prompt
    assert "pass around_move=N" in prompt
    assert "where did I go wrong?" in prompt
    assert "what was my biggest mistake?" in prompt


def test_find_eval_swings_is_registered_for_gemini():
    assert tools.find_eval_swings in tools.ALL_TOOLS


def test_explain_position_is_registered_for_gemini():
    assert tools.explain_position in tools.ALL_TOOLS


def test_system_prompt_routes_conceptual_questions_to_explain_position():
    game = chess.pgn.read_game(io.StringIO("1. e4 e5 1/2-1/2"))
    assert game is not None
    prompt = build_system_prompt(SessionState.from_game(game))
    assert "call explain_position()" in prompt
    assert "what are the imbalances?" in prompt


# ---- player-aware prompt + colour selection -------------------------------


_NAMED = '[White "Alice"]\n[Black "Bob"]\n[Result "0-1"]\n\n1. e4 e5 0-1'


def test_system_prompt_states_the_users_colour_and_name():
    prompt = build_system_prompt(_session(_NAMED, player_color="black"))
    assert "Bob" in prompt  # the user
    assert "played the black pieces" in prompt
    assert "Alice" in prompt  # the opponent
    assert "your opponent" in prompt


def test_system_prompt_unknown_colour_tells_model_to_ask():
    prompt = build_system_prompt(_session(_NAMED))  # colour not chosen
    assert "do NOT yet know which side the user played" in prompt
    assert "never infer it from who won" in prompt


def test_system_prompt_routes_questions_to_the_right_scope():
    prompt = build_system_prompt(_session(_NAMED, player_color="white"))
    assert 'scope="player"' in prompt
    assert 'scope="opponent"' in prompt
    assert 'scope="both"' in prompt


def test_prompt_player_color_reads_white_or_black():
    quiet = lambda *a, **k: None
    assert prompt_player_color(_session(_NAMED), ask=lambda _="": "white", show=quiet) == "white"
    assert prompt_player_color(_session(_NAMED), ask=lambda _="": "black", show=quiet) == "black"


def test_prompt_player_color_skip_or_eof_returns_none():
    quiet = lambda *a, **k: None
    assert prompt_player_color(_session(_NAMED), ask=lambda _="": "skip", show=quiet) is None

    def eof(_=""):
        raise EOFError

    assert prompt_player_color(_session(_NAMED), ask=eof, show=quiet) is None


# ---- investigation prompt ---------------------------------------------------


def test_system_prompt_teaches_the_model_to_investigate_with_probes():
    prompt = build_system_prompt(_session(_NAMED))
    assert "INVESTIGATE like a coach" in prompt
    for probe in ('try_line("', 'square_info("', "threats()"):
        assert probe in prompt
    assert "Name an idea ONLY" in prompt
    assert "Probes never change the verdict" in prompt


# ---- the claim check: moves in the answer must come from a tool ------------


def test_move_mentions_finds_piece_moves_captures_castling_and_promotions():
    text = "Nxe5 wins, then Qxe5 and O-O-O; exd5 and e8=Q too. The pawn on e4 is a square."
    assert set(tutor.move_mentions(text).values()) == {"Nxe5", "Qxe5", "O-O-O", "exd5", "e8=Q"}


def test_move_mentions_ignores_plain_pawn_pushes_and_words():
    assert tutor.move_mentions("Play e4 on move 1 — Books and Queens are words.") == {}


def test_unverified_moves_tolerates_spelling_differences():
    grounded = set(tutor.move_mentions("Nbd2 Qxh5+"))
    assert tutor.unverified_moves("Nd2 then Qh5 is fine.", grounded) == []
    assert tutor.unverified_moves("Nd2, then Bb5+!", grounded) == ["Bb5"]


def test_configured_model_defaults_and_reads_the_environment(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert tutor.configured_model() == tutor.DEFAULT_MODEL
    monkeypatch.setenv("GEMINI_MODEL", "  some-stronger-model ")
    assert tutor.configured_model() == "some-stronger-model"


class _FakeGemini:
    """Stands in for `genai.Client`: replays scripted model turns, records requests."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.requests = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.requests.append({"model": model, "contents": list(contents)})
        return self._turns.pop(0)


def _says(text):
    content = types.Content(role="model", parts=[types.Part.from_text(text=text)])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)], text=text)


def _calls(name):
    part = types.Part(function_call=types.FunctionCall(name=name, args={}))
    content = types.Content(role="model", parts=[part])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)], text=None)


def _probe() -> dict:
    """A stand-in tool that 'returns' Nxe5 from the engine."""
    return {"ok": True, "best": "Nxe5"}


def _tutor(turns, known_moves=()):
    fake = _FakeGemini(turns)
    adapter = tutor.GeminiTutor(
        "system", [_probe], api_key="unused", known_moves=known_moves,
        model="test-model", client=fake,
    )
    return adapter, fake


def _last_user_text(fake):
    last = fake.requests[-1]["contents"][-1]
    return last.parts[0].text


def test_answer_naming_only_tool_returned_moves_passes_unchanged():
    adapter, fake = _tutor([_calls("_probe"), _says("Nxe5 wins a pawn.")])
    assert adapter.ask("what's best?") == "Nxe5 wins a pawn."
    assert len(fake.requests) == 2  # tool round + answer; no correction round
    assert fake.requests[0]["model"] == "test-model"


def test_game_moves_and_the_users_own_moves_count_as_grounded():
    adapter, fake = _tutor([_says("Your Bc4 was fine, and Qh5 is legal.")], known_moves=["Bc4"])
    assert adapter.ask("was Qh5 possible?") == "Your Bc4 was fine, and Qh5 is legal."
    assert len(fake.requests) == 1


def test_unverified_move_triggers_one_rewrite():
    adapter, fake = _tutor([_says("Qh5 is the best move."), _says("The engine prefers Nxe5.")],
                           known_moves=["Nxe5"])
    assert adapter.ask("what's best?") == "The engine prefers Nxe5."
    assert len(fake.requests) == 2
    correction = _last_user_text(fake)
    assert "Qh5" in correction and "no tool returned" in correction


def test_unverified_move_that_survives_the_rewrite_gets_a_visible_caution():
    adapter, _ = _tutor([_says("Qh5 is best."), _says("Still, Qh5 is best.")])
    answer = adapter.ask("what's best?")
    assert answer.startswith("Still, Qh5 is best.")
    assert "Caution: Qh5 did not come from the engine tools" in answer


def _nested_probe() -> dict:
    """Returns a move after a newline, nested in a list — must still be grounded."""
    return {"ok": True, "lines": [{"note": "engine line:\nNf3 then Bb5"}]}


def test_moves_nested_or_after_newlines_in_tool_results_are_grounded():
    fake = _FakeGemini([_calls("_nested_probe"), _says("Nf3 and then Bb5 is the plan.")])
    adapter = tutor.GeminiTutor("system", [_nested_probe], api_key="unused",
                                model="test-model", client=fake)
    assert adapter.ask("plan?") == "Nf3 and then Bb5 is the plan."
    assert len(fake.requests) == 2  # no correction round
