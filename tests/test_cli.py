"""The CLI's entry point and its .env loader.

``main`` reads ``.env`` before it parses a single argument, so anything that goes
wrong in the loader breaks every command, not just the ones that need a key.
These tests pin the loader's contract and the exit codes ``main`` promises.

Every test replaces ``os.environ`` with a throwaway dict. The repository root can
hold a real ``.env``, and the loader writes straight into the environment, so
without this a test run would leak keys into whatever runs after it.
"""

from __future__ import annotations

import argparse
import os

import pytest

from jev_xray import cli


@pytest.fixture
def env(monkeypatch):
    fresh: dict[str, str] = {}
    monkeypatch.setattr(os, "environ", fresh)
    return fresh


def loaded(env: dict[str, str]) -> dict[str, str]:
    """What the loader put in the environment, without pytest's own bookkeeping."""
    return {k: v for k, v in env.items() if k != "PYTEST_CURRENT_TEST"}


def dotenv(tmp_path, content: str | bytes):
    path = tmp_path / ".env"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


class TestLoadDotenv:
    def test_sets_variables_from_the_file(self, tmp_path, env):
        cli._load_dotenv(dotenv(tmp_path, "FIRST=one\nSECOND=two\n"))
        assert loaded(env) == {"FIRST": "one", "SECOND": "two"}

    def test_an_existing_variable_wins_over_the_file(self, tmp_path, env):
        env["KEY"] = "exported"
        cli._load_dotenv(dotenv(tmp_path, "KEY=from-file\nOTHER=x\n"))
        assert loaded(env) == {"KEY": "exported", "OTHER": "x"}

    def test_comments_blank_lines_and_lines_without_equals_are_skipped(self, tmp_path, env):
        cli._load_dotenv(dotenv(tmp_path, "# a comment\n\n   \nnot a pair\nREAL=1\n"))
        assert loaded(env) == {"REAL": "1"}

    def test_matching_quotes_are_stripped(self, tmp_path, env):
        cli._load_dotenv(dotenv(tmp_path, "DOUBLE=\"a b\"\nSINGLE='c d'\nBARE=e\n"))
        assert loaded(env) == {"DOUBLE": "a b", "SINGLE": "c d", "BARE": "e"}

    def test_only_the_first_equals_sign_splits(self, tmp_path, env):
        cli._load_dotenv(dotenv(tmp_path, "URL=https://host/path?a=1&b=2\n"))
        assert loaded(env) == {"URL": "https://host/path?a=1&b=2"}

    def test_surrounding_whitespace_is_trimmed(self, tmp_path, env):
        cli._load_dotenv(dotenv(tmp_path, "  SPACED  =  value  \n"))
        assert loaded(env) == {"SPACED": "value"}

    def test_an_empty_value_sets_nothing(self, tmp_path, env):
        cli._load_dotenv(dotenv(tmp_path, "EMPTY=\nQUOTED=''\nFILLED=1\n"))
        assert loaded(env) == {"FILLED": "1"}

    def test_a_missing_file_is_ignored(self, tmp_path, env):
        cli._load_dotenv(tmp_path / "absent.env")
        assert loaded(env) == {}

    def test_a_directory_named_dotenv_is_ignored(self, tmp_path, env):
        (tmp_path / ".env").mkdir()
        cli._load_dotenv(tmp_path / ".env")
        assert loaded(env) == {}

    def test_a_file_that_is_not_utf8_is_ignored_not_fatal(self, tmp_path, env):
        # A .env saved as Latin-1 used to raise UnicodeDecodeError, which is not
        # an OSError, so every command died with a traceback. Nothing from the
        # file is applied: the read fails as a whole, before any line is parsed.
        cli._load_dotenv(dotenv(tmp_path, b"EARLY=1\nNOTE=caf\xe9\n"))
        assert loaded(env) == {}

    def test_defaults_to_the_dotenv_in_the_working_directory(self, tmp_path, env, monkeypatch):
        dotenv(tmp_path, "FROM_CWD=yes\n")
        monkeypatch.chdir(tmp_path)
        cli._load_dotenv()
        assert loaded(env) == {"FROM_CWD": "yes"}


class _Parser:
    """Stands in for ``build_parser()`` so ``main`` runs a function of our choosing."""

    def __init__(self, func):
        self._func = func

    def parse_args(self, argv):
        return argparse.Namespace(func=self._func)


@pytest.fixture
def run_main(monkeypatch):
    monkeypatch.setattr(cli, "_load_dotenv", lambda path=None: None)

    def run(func):
        monkeypatch.setattr(cli, "build_parser", lambda: _Parser(func))
        return cli.main([])

    return run


class TestMain:
    def test_returns_the_exit_code_of_the_command(self, run_main):
        assert run_main(lambda args: 0) == 0
        assert run_main(lambda args: 3) == 3

    def test_keyboard_interrupt_exits_130(self, run_main):
        def interrupted(args):
            raise KeyboardInterrupt

        assert run_main(interrupted) == 130

    def test_an_unexpected_error_is_one_line_on_stderr_and_exit_1(self, run_main, capsys):
        def broken(args):
            raise ValueError("boom")

        assert run_main(broken) == 1
        err = capsys.readouterr().err
        assert err.strip() == "jev-xray: ValueError: boom"

    def test_system_exit_passes_through_with_its_message(self, run_main):
        def refuses(args):
            raise SystemExit("no such question")

        with pytest.raises(SystemExit) as raised:
            run_main(refuses)
        assert raised.value.code == "no such question"

    def test_a_non_utf8_dotenv_does_not_stop_a_command(self, tmp_path, env, monkeypatch, capsys):
        # End to end: the real loader, the real parser, a real offline command.
        dotenv(tmp_path, b"NOTE=caf\xe9\n")
        monkeypatch.chdir(tmp_path)
        assert cli.main(["demo", "--no-color"]) == 0
        assert capsys.readouterr().out
