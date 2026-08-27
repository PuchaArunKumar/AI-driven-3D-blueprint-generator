"""Transformer-backed prompt understanding.

The encoder is optional, so tests that need it skip when it is unavailable;
the guardrail tests run either way, because they protect against damage the
semantic layer could do rather than depending on what it can add.
"""

from __future__ import annotations

import pytest

from app.pipeline import semantic
from app.pipeline.prompt_parser import build_view_prompt, parse_prompt
from app.schemas import ViewName

needs_encoder = pytest.mark.skipif(
    not semantic.available(), reason="sentence encoder unavailable"
)


class TestSourceHygiene:
    def test_no_literal_control_characters_in_patterns(self) -> None:
        """A regex \\b written as a raw backspace byte silently never matches.

        That exact mistake disabled both word-boundary guards once, so it is
        worth a test rather than a code review note.
        """
        from pathlib import Path

        source = Path(semantic.__file__).read_text(encoding="utf-8")
        assert chr(8) not in source, "literal backspace byte in source"
        assert "\\b" in source, "word-boundary escapes should be present"


class TestCanonicalisation:
    @needs_encoder
    @pytest.mark.parametrize(
        ("phrase", "expected"),
        [
            ("electric wheel cchair", "electric wheelchair"),   # typo + split word
            ("a electric wheel chair", "electric wheelchair"),  # split word
            ("wheelchiar", "wheelchair"),                       # transposition
        ],
    )
    def test_broken_object_phrases_are_repaired(self, phrase: str, expected: str) -> None:
        canonical, category, score = semantic.canonicalise_object(phrase)
        assert canonical == expected
        assert category is not None
        assert score >= semantic.OBJECT_THRESHOLD

    @needs_encoder
    @pytest.mark.parametrize(
        "phrase",
        [
            "bike frame",              # already names its head noun
            "sleek sports car",        # already exact
            "robot arm for a factory", # descriptive context must survive
            "stool",
            "wheel",                   # a wheel really is a wheel
        ],
    )
    def test_clear_phrases_are_left_alone(self, phrase: str) -> None:
        canonical, _, _ = semantic.canonicalise_object(phrase)
        assert canonical == phrase

    @needs_encoder
    @pytest.mark.parametrize("phrase", ["quantum flux capacitor", "cup", "grip aid"])
    def test_unlisted_objects_are_not_forced_onto_a_category(self, phrase: str) -> None:
        """Mislabelling a novel object is worse than leaving it alone."""
        canonical, _, _ = semantic.canonicalise_object(phrase)
        assert canonical == phrase

    def test_empty_input_is_safe(self) -> None:
        assert semantic.canonicalise_object("")[1] is None
        assert semantic.canonicalise_object("   ")[1] is None

    def test_character_ratio_ignores_spacing(self) -> None:
        """"wheel chair" must score against "wheelchair" like a plain typo."""
        assert semantic._character_ratio("wheel chair", "wheelchair") > 0.9
        assert semantic._character_ratio("banana", "wheelchair") < 0.4


class TestParserIntegration:
    @needs_encoder
    def test_a_typo_no_longer_reaches_the_image_prompt(self) -> None:
        """The bug this layer exists for: "wheel cchair" made the model draw a wheel."""
        spec = parse_prompt("a electric wheel cchair")
        assert spec.object == "electric wheelchair"

        prompt = build_view_prompt(spec, ViewName.FRONT)
        assert "wheelchair" in prompt
        assert "cchair" not in prompt

    @needs_encoder
    def test_repairing_the_object_reveals_hidden_vocabulary(self) -> None:
        """"wheelchair" only becomes matchable once the typo is fixed."""
        spec = parse_prompt("a electric wheel cchair")
        assert "wheelchair compatible" in spec.accessibility_requirements

    @needs_encoder
    def test_paraphrases_are_understood(self) -> None:
        """No regex would catch "cannot grip well"."""
        spec = parse_prompt("a kitchen tool for someone who cannot grip well")
        assert spec.accessibility_requirements

    def test_rule_matches_are_preserved(self) -> None:
        """The semantic pass augments; it must never drop a confident rule hit."""
        spec = parse_prompt("A stainless steel bracket, CNC machined, 40 x 20 x 5 mm")
        assert "stainless steel" in spec.materials
        assert "CNC machining" in spec.manufacturing_requirements
        assert spec.dimensions.length_mm == 40.0

    def test_parsing_stays_deterministic(self) -> None:
        prompt = "a electric wheel cchair for outdoor use"
        assert parse_prompt(prompt).model_dump() == parse_prompt(prompt).model_dump()

    def test_disabling_the_layer_leaves_the_rules_untouched(self, monkeypatch) -> None:
        """SEMANTIC_PROMPT=false must fall back cleanly, not error."""
        from app.config import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "semantic_prompt", False)
        spec = parse_prompt("a electric wheel cchair")
        assert spec.object == "electric wheel cchair"
