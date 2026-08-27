"""Prompt understanding: natural language -> DesignSpec."""

from __future__ import annotations

import pytest

from app.errors import ValidationError
from app.pipeline.prompt_parser import (
    build_view_prompt,
    parse_prompt,
    spec_to_base_prompt,
)
from app.schemas import ViewName


class TestValidation:
    @pytest.mark.parametrize("prompt", ["", "   ", "\n\t "])
    def test_empty_prompt_is_rejected(self, prompt: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            parse_prompt(prompt)
        assert "describe" in str(excinfo.value).lower() or "enter" in str(excinfo.value).lower()

    def test_too_short_prompt_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            parse_prompt("a")

    def test_none_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            parse_prompt(None)  # type: ignore[arg-type]

    def test_minimal_valid_prompt(self) -> None:
        spec = parse_prompt("cup")
        assert spec.object == "cup"


class TestObjectExtraction:
    @pytest.mark.parametrize(
        ("prompt", "expected"),
        [
            ("Design a wooden chair", "wooden chair"),
            ("Create an aluminium bracket", "aluminium bracket"),
            ("A modern desk lamp, brushed steel", "desk lamp"),
            ("Generate me a bike frame for commuting", "bike frame"),
            ("build the prototype housing with vents", "prototype housing"),
        ],
    )
    def test_lead_verbs_and_articles_are_stripped(self, prompt: str, expected: str) -> None:
        assert parse_prompt(prompt).object == expected

    def test_object_stops_at_clause_break(self) -> None:
        spec = parse_prompt("An electric car, carbon fibre body, 4.5m long")
        assert spec.object == "electric car"

    def test_leading_style_and_colour_move_to_their_own_fields(self) -> None:
        """They are re-added by spec_to_base_prompt, so keeping them would duplicate."""
        spec = parse_prompt("A futuristic matte black electric car")
        assert spec.object == "electric car"
        assert spec.style == "futuristic"
        assert spec.color == "matte black"
        assert spec_to_base_prompt(spec).startswith("matte black futuristic electric car")


class TestDimensions:
    def test_triple_with_trailing_unit(self) -> None:
        dimensions = parse_prompt("A box 40 x 30 x 20 cm").dimensions
        assert (dimensions.length_mm, dimensions.width_mm, dimensions.height_mm) == (
            400.0,
            300.0,
            200.0,
        )

    def test_triple_with_per_value_units(self) -> None:
        dimensions = parse_prompt("A tray 45cm x 30cm x 2cm").dimensions
        assert dimensions.length_mm == 450.0
        assert dimensions.height_mm == 20.0

    @pytest.mark.parametrize(
        ("prompt", "field", "expected"),
        [
            ("A shelf 1.2 m tall", "height_mm", 1200.0),
            ("A panel 500 mm wide", "width_mm", 500.0),
            ("A beam 6 feet long", "length_mm", 1828.8),
            ("A plate 12 inches wide", "width_mm", 304.8),
            ("A stand with a height of 75 cm", "height_mm", 750.0),
        ],
    )
    def test_named_axes_and_unit_conversion(
        self, prompt: str, field: str, expected: float
    ) -> None:
        assert getattr(parse_prompt(prompt).dimensions, field) == pytest.approx(expected)

    def test_weight_is_converted_to_grams(self) -> None:
        assert parse_prompt("A frame weighing 1.2 kg").dimensions.weight_g == 1200.0

    def test_load_rating_is_not_read_as_weight(self) -> None:
        """"must support 300 kg" is a capacity, not the product's own mass."""
        spec = parse_prompt("A ramp that must support 300 kg")
        assert spec.dimensions.weight_g is None

    def test_weight_after_a_load_clause_is_still_found(self) -> None:
        spec = parse_prompt("Shelf that holds 50 kg, weight 3.5 kg")
        assert spec.dimensions.weight_g == 3500.0


class TestVocabularies:
    def test_materials_prefer_the_longest_match(self) -> None:
        assert parse_prompt("A stainless steel pan").materials == ["stainless steel"]

    def test_material_spellings_are_normalised(self) -> None:
        assert parse_prompt("An aluminum panel").materials == ["aluminium"]
        assert parse_prompt("A carbon fiber frame").materials == ["carbon fibre"]

    def test_pla_does_not_match_inside_plastic(self) -> None:
        assert "PLA" not in parse_prompt("A plastic housing").materials

    @pytest.mark.parametrize(
        ("prompt", "expected"),
        [
            ("A 3D printable bracket", "additive manufacturing (3D printing)"),
            ("A 3D printed bracket", "additive manufacturing (3D printing)"),
            ("A CNC machined block", "CNC machining"),
            ("An injection moulded cover", "injection moulding"),
            ("A laser cut panel", "laser cutting"),
        ],
    )
    def test_manufacturing_matches_inflected_forms(self, prompt: str, expected: str) -> None:
        assert expected in parse_prompt(prompt).manufacturing_requirements

    def test_accessibility_terms_are_detected(self) -> None:
        spec = parse_prompt("A wheelchair tray for elderly users with arthritis")
        assert "wheelchair compatible" in spec.accessibility_requirements
        assert "low grip-strength operation" in spec.accessibility_requirements

    def test_constraints_are_detected(self) -> None:
        spec = parse_prompt("A foldable waterproof outdoor lightweight crate")
        for expected in ("foldable", "waterproof", "outdoor use", "lightweight"):
            assert expected in spec.constraints

    def test_style_and_colour(self) -> None:
        spec = parse_prompt("A minimalist matte black speaker")
        assert spec.style == "minimalist"
        assert spec.color == "matte black"

    def test_tolerance_is_parsed(self) -> None:
        assert parse_prompt("A gear with tolerance of 0.05 mm").tolerances_mm == 0.05


class TestPurpose:
    def test_purpose_follows_for(self) -> None:
        assert parse_prompt("A stand for a laptop").purpose == "a laptop"

    def test_purpose_stops_at_a_clause_break(self) -> None:
        spec = parse_prompt("A chair for a small apartment, 45 x 45 x 90 cm")
        assert spec.purpose == "a small apartment"


class TestPromptBuilding:
    def test_base_prompt_includes_identity_details(self) -> None:
        spec = parse_prompt("A minimalist matte black oak stool, 40 x 40 x 45 cm")
        base = spec_to_base_prompt(spec)
        assert "matte black" in base
        assert "minimalist" in base
        assert "oak" in base

    def test_every_view_gets_a_distinct_prompt(self) -> None:
        spec = parse_prompt("A red bicycle helmet")
        prompts = {view: build_view_prompt(spec, view) for view in ViewName}
        assert len(set(prompts.values())) == len(ViewName)
        # The shared identity clause must survive into all of them.
        assert all("red" in prompt for prompt in prompts.values())

    def test_view_prompt_names_its_camera_position(self) -> None:
        spec = parse_prompt("A red bicycle helmet")
        assert "top-down" in build_view_prompt(spec, ViewName.TOP)
        assert "front" in build_view_prompt(spec, ViewName.FRONT)


class TestDeterminism:
    def test_parsing_is_deterministic(self) -> None:
        prompt = "A foldable aluminium wheelchair ramp for outdoor use, 1.5 m long"
        assert parse_prompt(prompt).model_dump() == parse_prompt(prompt).model_dump()
