"""Tests for the tutor prompt and adapter-adjacent behavior."""

import io

import chess.pgn

import tools
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
