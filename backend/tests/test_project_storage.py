"""Project persistence, path safety and the asset layout."""

from __future__ import annotations

import uuid

import pytest

from app.errors import NotFoundError, ValidationError
from app.pipeline.prompt_parser import parse_prompt
from app.schemas import ProjectStatus, ViewName
from app.storage import (
    ensure_within,
    images_dir,
    project_dir,
    relative_url,
    repository,
    sanitise_filename,
    validate_project_id,
)


class TestCreateAndRead:
    def test_create_returns_a_persisted_project(self) -> None:
        created = repository.create("A walnut stool", design_spec=parse_prompt("A walnut stool"))
        fetched = repository.get(created.id)
        assert fetched.id == created.id
        assert fetched.prompt == "A walnut stool"
        assert fetched.design_spec is not None
        assert fetched.design_spec.materials == ["walnut"]

    def test_status_reflects_whether_a_spec_was_supplied(self) -> None:
        with_spec = repository.create("A stool", design_spec=parse_prompt("A stool"))
        without = repository.create("A stool")
        assert with_spec.status is ProjectStatus.SPECIFIED
        assert without.status is ProjectStatus.DRAFT

    def test_name_is_derived_from_the_prompt(self) -> None:
        project = repository.create("a foldable aluminium wheelchair ramp for outdoor use")
        assert project.name.startswith("A foldable aluminium")
        assert len(project.name) <= 200

    def test_missing_project_raises_not_found(self) -> None:
        with pytest.raises(NotFoundError):
            repository.get(str(uuid.uuid4()))

    def test_list_is_newest_first(self) -> None:
        first = repository.create("First design")
        second = repository.create("Second design")
        listed = [item.id for item in repository.list()]
        assert listed.index(second.id) < listed.index(first.id)

    def test_count_tracks_creation(self) -> None:
        before = repository.count()
        repository.create("Another design")
        assert repository.count() == before + 1

    def test_demo_projects_can_be_excluded(self) -> None:
        repository.create("A demo design", is_demo=True)
        repository.create("A user design")
        assert all(not item.is_demo for item in repository.list(include_demo=False))


class TestUpdate:
    def test_update_changes_fields_and_bumps_timestamp(self) -> None:
        project = repository.create("A stool")
        updated = repository.update(project.id, name="Renamed stool")
        assert updated.name == "Renamed stool"
        assert updated.updated_at >= project.updated_at

    def test_design_spec_roundtrips_through_json(self) -> None:
        project = repository.create("A stool")
        spec = parse_prompt("A matte black steel stool, 40 x 40 x 45 cm")
        updated = repository.update(project.id, design_spec=spec)
        assert updated.design_spec is not None
        assert updated.design_spec.color == "matte black"
        assert updated.design_spec.dimensions.height_mm == 450.0

    def test_unknown_fields_are_ignored(self) -> None:
        project = repository.create("A stool")
        repository.update(project.id, not_a_column="value")  # must not raise

    def test_history_accumulates_in_order(self) -> None:
        project = repository.create("A stool")
        repository.append_history(project.id, "first", "one")
        repository.append_history(project.id, "second", "two")
        history = repository.get(project.id).history
        assert [entry["event"] for entry in history] == ["first", "second"]
        assert history[0]["detail"] == "one"

    def test_set_images_stores_and_orders_views(self) -> None:
        from datetime import datetime, timezone

        from app.schemas import GeneratedImage

        project = repository.create("A stool")
        now = datetime.now(timezone.utc)
        images = [
            GeneratedImage(view=ViewName.TOP, path="/tmp/top.png", url="", width=1,
                           height=1, provider="test", created_at=now),
            GeneratedImage(view=ViewName.FRONT, path="/tmp/front.png", url="", width=1,
                           height=1, provider="test", created_at=now),
        ]
        stored = repository.set_images(project.id, images)
        assert len(stored.images) == 2
        assert {image.view for image in stored.images} == {ViewName.TOP, ViewName.FRONT}


class TestDelete:
    def test_delete_removes_the_project(self) -> None:
        project = repository.create("A stool")
        repository.delete(project.id)
        with pytest.raises(NotFoundError):
            repository.get(project.id)

    def test_delete_removes_the_asset_directory(self) -> None:
        project = repository.create("A stool")
        directory = project_dir(project.id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.glb").write_bytes(b"data")
        repository.delete(project.id)
        assert not directory.exists()

    def test_deleting_twice_raises_not_found(self) -> None:
        project = repository.create("A stool")
        repository.delete(project.id)
        with pytest.raises(NotFoundError):
            repository.delete(project.id)


class TestPathSafety:
    @pytest.mark.parametrize(
        "bad_id",
        [
            "../../etc/passwd",
            "..\\..\\windows\\system32",
            "not-a-uuid",
            "",
            "'; DROP TABLE projects;--",
            "../" * 10,
        ],
    )
    def test_invalid_project_ids_are_rejected(self, bad_id: str) -> None:
        with pytest.raises(ValidationError):
            validate_project_id(bad_id)

    def test_valid_uuid_is_accepted_and_lowercased(self) -> None:
        identifier = str(uuid.uuid4()).upper()
        assert validate_project_id(identifier) == identifier.lower()

    def test_project_dir_rejects_traversal(self) -> None:
        with pytest.raises(ValidationError):
            project_dir("../escape")

    @pytest.mark.parametrize(
        ("raw", "expected_absent"),
        [
            ("../../evil.png", ".."),
            ("C:\\Windows\\system32\\evil.png", "\\"),
            ("/etc/passwd", "/"),
        ],
    )
    def test_filenames_are_stripped_of_paths(self, raw: str, expected_absent: str) -> None:
        cleaned = sanitise_filename(raw)
        assert expected_absent not in cleaned

    def test_sanitise_falls_back_for_empty_input(self) -> None:
        assert sanitise_filename("...") == "file"

    def test_ensure_within_blocks_escape(self, storage_root) -> None:
        with pytest.raises(ValidationError):
            ensure_within(storage_root / ".." / ".." / "secrets.txt", storage_root)

    def test_ensure_within_allows_a_child(self, storage_root) -> None:
        assert ensure_within(storage_root / "models" / "a.glb", storage_root)


class TestUrlMapping:
    def test_paths_inside_storage_become_urls(self, storage_root) -> None:
        url = relative_url(storage_root / "models" / "abc" / "model.glb")
        assert url == "/files/models/abc/model.glb"

    def test_paths_outside_storage_have_no_url(self) -> None:
        assert relative_url("C:/Windows/system32/config") is None

    def test_empty_path_has_no_url(self) -> None:
        assert relative_url("") is None

    def test_images_dir_is_inside_the_project_dir(self) -> None:
        identifier = str(uuid.uuid4())
        assert project_dir(identifier) in images_dir(identifier).parents
