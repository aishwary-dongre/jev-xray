"""The example fixtures.

Most of them need a live provider to run, so CI never exercises them. This is
the offline check that they still load the way the CLI loads them, that every
question parses into a real question type, and that the command quoted in each
``_comment`` names a question (and a trusted-boundary field) that exists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from jev_xray import question_from_wire
from jev_xray.cli import _load_input

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.json"))


def test_examples_are_found():
    assert EXAMPLES, "no example fixtures found; the glob is pointing at the wrong place"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
class TestExampleFixtures:
    def test_loads_the_way_the_cli_loads_it(self, path):
        state, questions = _load_input(path)
        assert state
        assert questions

    def test_every_question_parses(self, path):
        _, questions = _load_input(path)
        for qid, body in questions.items():
            assert question_from_wire(body), qid

    def test_comment_names_questions_that_exist(self, path):
        _, questions = _load_input(path)
        comment = json.loads(path.read_text(encoding="utf-8")).get("_comment", "")
        for qid in re.findall(r"(?:-q|--question)\s+(\w+)", comment):
            assert qid in questions, f"{path.name} _comment runs -q {qid}, which it does not define"

    def test_comment_names_untrusted_fields_that_exist(self, path):
        state, _ = _load_input(path)
        comment = json.loads(path.read_text(encoding="utf-8")).get("_comment", "")
        for field in re.findall(r"--untrusted\s+'([^']+)'", comment):
            assert isinstance(state, dict), (
                f"{path.name} marks {field!r} untrusted in a non-object state"
            )
            assert field in state, (
                f"{path.name} _comment marks {field!r} untrusted, which is not a state field"
            )
