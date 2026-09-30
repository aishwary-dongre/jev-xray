"""Segmentation and ablation.

Segments are the vocabulary every explanation is written in, so the invariants
here are load bearing: spans must index back into the original text exactly, and
ablation must produce a state that still reads naturally. A stray double space
is not cosmetic when the reader is documented to be literal.
"""

from __future__ import annotations

import pytest

from jev_xray import (
    JsonFieldSegmenter,
    LineSegmenter,
    SentenceSegmenter,
    TurnSegmenter,
    auto_segmenter,
    get_segmenter,
)
from jev_xray.segment import DEFAULT_MASK

PROSE = (
    "The charge appeared twice on my card. "
    "I have already emailed once about it. "
    "Please refund the duplicate."
)


class TestSentenceSegmenter:
    def test_splits_on_sentence_boundaries(self):
        segments = SentenceSegmenter().split(PROSE)
        assert len(segments) == 3
        assert segments[0].text == "The charge appeared twice on my card."
        assert segments[2].text == "Please refund the duplicate."

    def test_spans_index_back_into_the_original(self):
        for segment in SentenceSegmenter().split(PROSE):
            assert PROSE[segment.start : segment.end] == segment.text

    def test_ids_are_positional(self):
        assert [s.id for s in SentenceSegmenter().split(PROSE)] == [0, 1, 2]

    def test_delete_removes_the_span_and_its_trailing_space(self):
        result = SentenceSegmenter().ablate(PROSE, [1])
        assert "already emailed" not in result
        assert "The charge appeared twice on my card. Please refund" in result
        assert "  " not in result

    def test_mask_preserves_position(self):
        result = SentenceSegmenter().ablate(PROSE, [1], mode="mask")
        assert "already emailed" not in result
        assert DEFAULT_MASK in result
        assert result.startswith("The charge appeared twice")
        assert result.endswith("Please refund the duplicate.")

    def test_dropping_everything_leaves_nothing_meaningful(self):
        result = SentenceSegmenter().ablate(PROSE, [0, 1, 2])
        assert result.strip() == ""

    def test_multiple_drops_do_not_corrupt_offsets(self):
        result = SentenceSegmenter().ablate(PROSE, [0, 2])
        assert result.strip() == "I have already emailed once about it."

    def test_empty_drop_is_the_identity(self):
        assert SentenceSegmenter().ablate(PROSE, []) == PROSE

    def test_unknown_segment_id_is_an_error(self):
        with pytest.raises(KeyError, match="no such segment"):
            SentenceSegmenter().ablate(PROSE, [99])

    def test_blank_lines_also_break_segments(self):
        segments = SentenceSegmenter().split("First block\n\nSecond block")
        assert [s.text for s in segments] == ["First block", "Second block"]

    def test_rejects_a_structured_state(self):
        with pytest.raises(TypeError, match="needs a text state"):
            SentenceSegmenter().split({"a": "b"})


class TestLineSegmenter:
    LOG = "INFO boot\n\nWARN disk at 91%\nERROR write failed\n"

    def test_skips_blank_lines(self):
        segments = LineSegmenter().split(self.LOG)
        assert [s.text for s in segments] == [
            "INFO boot",
            "WARN disk at 91%",
            "ERROR write failed",
        ]

    def test_spans_index_back_into_the_original(self):
        for segment in LineSegmenter().split(self.LOG):
            assert self.LOG[segment.start : segment.end] == segment.text

    def test_ablation_drops_the_line(self):
        result = LineSegmenter().ablate(self.LOG, [1])
        assert "disk at 91%" not in result
        assert "ERROR write failed" in result


class TestTurnSegmenter:
    CHAT = (
        "Customer: I was charged twice.\n"
        "It happened on the 3rd.\n"
        "Agent: Let me take a look.\n"
        "Customer: Thanks."
    )

    def test_a_turn_keeps_its_continuation_lines(self):
        segments = TurnSegmenter().split(self.CHAT)
        assert len(segments) == 3
        assert "It happened on the 3rd." in segments[0].text
        assert segments[1].text.startswith("Agent:")

    def test_falls_back_to_lines_without_speakers(self):
        segments = TurnSegmenter().split("no speaker here\nsecond line")
        assert [s.text for s in segments] == ["no speaker here", "second line"]

    def test_text_before_the_first_speaker_is_kept(self):
        segments = TurnSegmenter().split("Transcript follows.\nAlice: hello")
        assert len(segments) == 2
        assert segments[0].text == "Transcript follows."


class TestJsonFieldSegmenter:
    STATE = {
        "ticket": {"subject": "duplicate charge", "body": "I was billed twice."},
        "messages": ["first note", "second note", "third note"],
        "priority": 3,
    }

    def test_paths_use_dot_and_index_form(self):
        paths = [s.path for s in JsonFieldSegmenter().split(self.STATE)]
        assert "ticket.subject" in paths
        assert "messages.1" in paths

    def test_scalars_are_excluded_by_default(self):
        paths = [s.path for s in JsonFieldSegmenter().split(self.STATE)]
        assert "priority" not in paths

    def test_scalars_can_be_included(self):
        paths = [s.path for s in JsonFieldSegmenter(include_scalars=True).split(self.STATE)]
        assert "priority" in paths

    def test_delete_removes_the_key(self):
        segmenter = JsonFieldSegmenter()
        target = next(s for s in segmenter.split(self.STATE) if s.path == "ticket.subject")
        result = segmenter.ablate(self.STATE, [target.id])
        assert "subject" not in result["ticket"]
        assert result["ticket"]["body"] == "I was billed twice."

    def test_mask_replaces_the_value(self):
        segmenter = JsonFieldSegmenter()
        target = next(s for s in segmenter.split(self.STATE) if s.path == "ticket.subject")
        result = segmenter.ablate(self.STATE, [target.id], mode="mask")
        assert result["ticket"]["subject"] == DEFAULT_MASK

    def test_list_deletions_do_not_shift_each_other(self):
        segmenter = JsonFieldSegmenter()
        segments = {s.path: s.id for s in segmenter.split(self.STATE)}
        result = segmenter.ablate(self.STATE, [segments["messages.0"], segments["messages.1"]])
        assert result["messages"] == ["third note"]

    def test_the_original_state_is_never_mutated(self):
        segmenter = JsonFieldSegmenter()
        target = segmenter.split(self.STATE)[0]
        segmenter.ablate(self.STATE, [target.id])
        assert self.STATE["ticket"]["subject"] == "duplicate charge"
        assert len(self.STATE["messages"]) == 3

    def test_rejects_a_text_state(self):
        with pytest.raises(TypeError, match="needs an object or array"):
            JsonFieldSegmenter().split("plain text")


class TestSelection:
    def test_registry_resolves_names(self):
        assert isinstance(get_segmenter("sentences"), SentenceSegmenter)
        assert isinstance(get_segmenter("line"), LineSegmenter)
        assert isinstance(get_segmenter("json"), JsonFieldSegmenter)

    def test_an_instance_passes_through(self):
        instance = LineSegmenter()
        assert get_segmenter(instance) is instance

    def test_unknown_name_lists_the_options(self):
        with pytest.raises(ValueError, match="unknown segmenter"):
            get_segmenter("paragraphs")

    def test_auto_picks_fields_for_structured_state(self):
        assert isinstance(auto_segmenter({"a": "b"}), JsonFieldSegmenter)

    def test_auto_picks_turns_for_a_transcript(self):
        assert isinstance(auto_segmenter("Alice: hi\nBob: hello"), TurnSegmenter)

    def test_auto_picks_sentences_for_prose(self):
        assert isinstance(auto_segmenter(PROSE), SentenceSegmenter)

    def test_auto_picks_lines_for_short_line_oriented_text(self):
        assert isinstance(auto_segmenter("one\ntwo\nthree\nfour"), LineSegmenter)


class TestSegmentPresentation:
    def test_preview_collapses_whitespace(self):
        segment = SentenceSegmenter().split("A  ragged\n   line here. Next one.")[0]
        assert segment.preview() == "A ragged line here."

    def test_preview_truncates_with_an_ellipsis(self):
        segment = SentenceSegmenter().split(PROSE)[0]
        preview = segment.preview(20)
        assert len(preview) == 20
        assert preview.endswith("\u2026")

    def test_preview_leaves_short_text_alone(self):
        segment = SentenceSegmenter().split("Short one. And another.")[0]
        assert segment.preview(50) == "Short one."

    def test_label_prefers_a_path_when_there_is_one(self):
        text_segment = SentenceSegmenter().split(PROSE)[0]
        assert text_segment.label == "sentence 0"
        field_segment = JsonFieldSegmenter().split({"a": {"b": "value"}})[0]
        assert field_segment.label == "a.b"


class TestEmptyContainerPruning:
    """Deleting the last leaf under a key should not leave the key behind.

    ``"customer": {}`` is an artifact of how the removal was done, not content the
    model was ever meant to read, and it still costs tokens. It also made the
    dead-weight output look wrong: pruning six of eight fields and reporting the
    state as 47% smaller, while the JSON still carried empty husks.
    """

    STATE = {
        "ticket": {"body": "I was billed twice."},
        "customer": {"tier": "gold"},
        "notes": ["only note"],
        "policy": "Duplicates are refundable.",
    }

    def _drop(self, paths, **kwargs):
        segmenter = JsonFieldSegmenter(**kwargs)
        ids = {s.path: s.id for s in segmenter.split(self.STATE)}
        return segmenter.ablate(self.STATE, [ids[p] for p in paths])

    def test_an_emptied_object_is_removed(self):
        result = self._drop(["customer.tier"])
        assert "customer" not in result

    def test_an_emptied_array_is_removed(self):
        result = self._drop(["notes.0"])
        assert "notes" not in result

    def test_surviving_content_is_untouched(self):
        result = self._drop(["customer.tier", "notes.0"])
        assert result["ticket"]["body"] == "I was billed twice."
        assert result["policy"] == "Duplicates are refundable."

    def test_nesting_is_pruned_bottom_up(self):
        state = {"a": {"b": {"c": "only leaf"}}, "keep": "yes"}
        segmenter = JsonFieldSegmenter()
        target = next(s for s in segmenter.split(state) if s.path == "a.b.c")
        result = segmenter.ablate(state, [target.id])
        # a.b emptied, so a empties, so a goes.
        assert result == {"keep": "yes"}

    def test_pruning_can_be_turned_off(self):
        result = self._drop(["customer.tier"], prune_empty=False)
        assert result["customer"] == {}

    def test_mask_mode_never_prunes(self):
        segmenter = JsonFieldSegmenter()
        ids = {s.path: s.id for s in segmenter.split(self.STATE)}
        result = segmenter.ablate(self.STATE, [ids["customer.tier"]], mode="mask")
        assert result["customer"]["tier"] == DEFAULT_MASK

    def test_scalars_and_empty_strings_survive(self):
        # Only containers emptied by our own edit are artifacts. An empty string
        # the caller supplied is content, however uninformative.
        state = {"blank": "", "count": 0, "body": "text here", "nothing": None}
        segmenter = JsonFieldSegmenter()
        target = next(s for s in segmenter.split(state) if s.path == "body")
        result = segmenter.ablate(state, [target.id])
        assert result == {"blank": "", "count": 0, "nothing": None}

    def test_the_original_state_is_still_not_mutated(self):
        self._drop(["customer.tier"])
        assert self.STATE["customer"] == {"tier": "gold"}
