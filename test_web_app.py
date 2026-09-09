"""Tests for the web layer — exercising `ReviewApp` directly, no HTTP needed.

`ReviewApp` returns plain dicts and takes no HTTP types, so the whole API surface
is testable without a running server. The board/navigation/colour paths need no
engine or API key; the `ask` guard paths are tested without either, proving the
server returns useful errors instead of crashing when they're missing.
"""

import web_app

# A named game with a known result, so we can assert player metadata flows through.
NAMED_PGN = '[White "Alice"]\n[Black "Bob"]\n[Result "0-1"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 0-1'


def app(engine=None, api_key=None) -> web_app.ReviewApp:
    return web_app.ReviewApp(engine=engine, api_key=api_key)


def test_state_before_any_load_reports_not_loaded():
    s = app().state()
    assert s["ok"] is True
    assert s["loaded"] is False
    assert s["engine_available"] is False
    assert s["api_key_present"] is False


def test_load_valid_pgn_returns_players_result_moves_and_board():
    a = app()
    s = a.load_game(pgn=NAMED_PGN)
    assert s["ok"] is True and s["loaded"] is True
    assert s["white"] == "Alice"
    assert s["black"] == "Bob"
    assert s["result"] == "0-1"
    assert [m["san"] for m in s["moves"]] == ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
    assert s["moves"][0] == {"ply": 1, "san": "e4", "move_number": 1, "side": "white"}
    assert s["moves"][1]["side"] == "black"
    assert s["current_ply"] == 0
    assert s["player_color"] is None  # never inferred from the result
    assert "<svg" in s["board_svg"]
    assert s["svg_error"] is None


def test_load_invalid_pgn_returns_error_not_crash():
    s = app().load_game(pgn="this is not a pgn")
    assert s["ok"] is False
    assert "Couldn't load" in s["error"]


def test_load_sample_game():
    s = app().load_game(sample=True)
    assert s["ok"] is True and s["loaded"] is True
    assert len(s["moves"]) > 0


def test_set_color_white_then_black():
    a = app()
    a.load_game(pgn=NAMED_PGN)
    assert a.set_color("white")["player_color"] == "white"
    assert a.set_color("black")["player_color"] == "black"


def test_set_color_invalid_returns_error():
    a = app()
    a.load_game(pgn=NAMED_PGN)
    result = a.set_color("grey")
    assert result["ok"] is False
    assert "white" in result["error"]  # the validation message names the valid options


def test_set_color_before_load_is_handled():
    result = app().set_color("white")
    assert result["ok"] is False
    assert "No game loaded" in result["error"]


def test_goto_updates_ply_and_board():
    a = app()
    a.load_game(pgn=NAMED_PGN)
    s = a.goto(4)  # after 1.e4 e5 2.Nf3 Nc6
    assert s["ok"] is True
    assert s["current_ply"] == 4
    assert s["side_to_move"] == "white"
    assert s["last_move_san"] == "Nc6"
    assert "<svg" in s["board_svg"]


def test_goto_out_of_range_returns_error():
    a = app()
    a.load_game(pgn=NAMED_PGN)
    result = a.goto(999)
    assert result["ok"] is False
    assert "out of range" in result["error"]


def test_goto_non_integer_is_handled():
    a = app()
    a.load_game(pgn=NAMED_PGN)
    result = a.goto("notanumber")
    assert result["ok"] is False
    assert "integer" in result["error"]


def test_ask_without_session_is_handled():
    result = app(api_key="fake").ask("where did I go wrong?")
    assert result["ok"] is False
    assert "No game loaded" in result["error"]


def test_ask_without_api_key_returns_useful_error_not_crash():
    a = app(api_key=None)
    a.load_game(pgn=NAMED_PGN)
    a.set_color("white")
    result = a.ask("where did I go wrong?")
    assert result["ok"] is False
    assert "GEMINI_API_KEY" in result["error"]


def test_ask_without_engine_returns_useful_error_not_crash():
    # API key present but no engine — must report the missing engine, not crash.
    a = app(api_key="fake", engine=None)
    a.load_game(pgn=NAMED_PGN)
    a.set_color("white")
    result = a.ask("where did I go wrong?")
    assert result["ok"] is False
    assert "Stockfish" in result["error"]


def test_ask_empty_question_is_handled():
    a = app(api_key="fake")
    a.load_game(pgn=NAMED_PGN)
    result = a.ask("   ")
    assert result["ok"] is False
    assert "Ask a question" in result["error"]
