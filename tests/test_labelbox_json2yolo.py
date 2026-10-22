# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for labelbox_json2yolo.py: loader variants and full conversion.

The existing suite only covers NDJSON loading; the JSON list / single
object / empty cases and the end-to-end convert() path are covered here.
"""

import json

import pytest
import yaml
from PIL import Image

from labelbox_json2yolo import convert, load_labelbox_json


def test_load_labelbox_json_list(tmp_path):
    file = tmp_path / "export.json"
    file.write_text(json.dumps([{"External ID": "a.jpg"}, {"External ID": "b.jpg"}]))
    assert load_labelbox_json(file) == [{"External ID": "a.jpg"}, {"External ID": "b.jpg"}]


def test_load_labelbox_single_object_is_wrapped_in_list(tmp_path):
    file = tmp_path / "export.json"
    file.write_text('{"External ID": "a.jpg"}')
    assert load_labelbox_json(file) == [{"External ID": "a.jpg"}]


@pytest.mark.parametrize("content", ["", "   ", "\n\n  \n"])
def test_load_labelbox_empty_file_returns_empty_list(tmp_path, content):
    file = tmp_path / "export.json"
    file.write_text(content)
    assert load_labelbox_json(file) == []


def test_load_labelbox_ndjson_with_blank_lines(tmp_path):
    file = tmp_path / "export.json"
    file.write_text('{"a": 1}\n\n{"b": 2}\n')
    assert load_labelbox_json(file) == [{"a": 1}, {"b": 2}]


def test_load_labelbox_scalar_json_is_wrapped(tmp_path):
    file = tmp_path / "export.json"
    file.write_text("42")
    assert load_labelbox_json(file) == [42]


@pytest.mark.parametrize("content", ['{"a": 1}\nnot json\n', "[1, 2,", "{broken}"])
def test_load_labelbox_invalid_content_raises(tmp_path, content):
    file = tmp_path / "export.json"
    file.write_text(content)
    with pytest.raises(json.JSONDecodeError):
        load_labelbox_json(file)


def test_convert_local_image_end_to_end(tmp_path, monkeypatch, capsys):
    # convert() builds its output dir from the file stem, relative to the CWD.
    monkeypatch.chdir(tmp_path)
    image = tmp_path / "photo.jpg"
    Image.new("RGB", (200, 100)).save(image)
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(
            [
                {
                    "Labeled Data": str(image),
                    "External ID": "photo.jpg",
                    "Label": {
                        "objects": [
                            {"bbox": {"top": 10, "left": 20, "height": 30, "width": 40}, "value": "cat"},
                            {"bbox": {"top": 0, "left": 0, "height": 100, "width": 200}, "value": "dog"},
                            {"bbox": {"top": 0, "left": 0, "height": 10, "width": 10}, "value": "cat"},
                            {"value": "no-bbox-object"},  # skipped with a warning
                        ]
                    },
                }
            ]
        )
    )

    convert(export, zip=False)

    save_dir = tmp_path / "export"
    lines = (save_dir / "labels" / "photo.txt").read_text().splitlines()
    # cat: center ((20+20)/200, (10+15)/100); dog: full image; cat again reuses id 0.
    assert lines == ["0 0.2 0.25 0.2 0.3", "1 0.5 0.5 1 1", "0 0.025 0.05 0.05 0.1"]
    assert (save_dir / "images" / "photo.jpg").exists()
    meta = yaml.safe_load((save_dir / "export.yaml").read_text())
    assert meta["nc"] == 2
    assert meta["names"] == ["cat", "dog"]  # first-seen order, not sorted
    assert "Skipping non-bbox" in capsys.readouterr().out


def test_convert_rejects_non_dict_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    export = tmp_path / "export.json"
    export.write_text(json.dumps([42]))
    with pytest.raises(TypeError, match="Expected Labelbox record dictionaries"):
        convert(export, zip=False)
