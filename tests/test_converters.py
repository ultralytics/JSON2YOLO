# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import base64
import io
import json
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from general_json2yolo import convert_coco_json, convert_labelme_json, convert_vott_json
from labelbox_json2yolo import load_labelbox_json


def test_coco_conversion_writes_nested_labels_and_yaml(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "instances_train.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "height": 100, "width": 200, "file_name": "nested/image1.jpg"}],
                "categories": [{"id": 1, "name": "person"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 20, 40, 30],
                        "segmentation": [[10, 20, 50, 20, 50, 50, 10, 50]],
                    }
                ],
            }
        )
    )

    save_dir = convert_coco_json(annotations, use_segments=True, save_dir=tmp_path / "out")

    label = (save_dir / "labels" / "train" / "nested" / "image1.txt").read_text().strip()
    assert label == "0 0.05 0.2 0.25 0.2 0.25 0.5 0.05 0.5"
    assert yaml.safe_load((save_dir / "instances_train.yaml").read_text()) == {"names": {0: "person"}}


def test_coco_conversion_writes_keypoints(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "person_keypoints_val.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "height": 100, "width": 200, "file_name": "image1.jpg"}],
                "categories": [{"id": 1, "name": "person"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "keypoints": [20, 30, 2, 40, 50, 2],
                    }
                ],
            }
        )
    )

    save_dir = convert_coco_json(annotations, use_keypoints=True, save_dir=tmp_path / "out")

    label = (save_dir / "labels" / "person_keypoints_val" / "image1.txt").read_text().strip().split()
    assert label == ["0", "0.15", "0.4", "0.1", "0.2", "0.1", "0.3", "2", "0.2", "0.5", "2"]


def test_labelme_conversion_writes_segments_boxes_and_yaml(tmp_path):
    image = tmp_path / "images" / "image1.jpg"
    image.parent.mkdir()
    image.write_bytes(b"fake image bytes")
    (tmp_path / "ann.json").write_text(
        json.dumps(
            {
                "imagePath": "images/image1.jpg",
                "imageHeight": 100,
                "imageWidth": 200,
                "shapes": [
                    {"label": "lane", "shape_type": "polygon", "points": [[10, 20], [50, 20], [50, 60]]},
                    {"label": "sign", "shape_type": "rectangle", "points": [[80, 10], [120, 30]]},
                ],
            }
        )
    )

    save_dir = convert_labelme_json(tmp_path, use_segments=True, save_dir=tmp_path / "out")

    lines = (save_dir / "labels" / "images" / "image1.txt").read_text().strip().splitlines()
    assert lines[0] == "0 0.05 0.2 0.25 0.2 0.25 0.6"
    assert lines[1] == "1 0.5 0.2 0.2 0.2"
    assert (save_dir / "images" / "images" / "image1.jpg").exists()
    assert yaml.safe_load((save_dir / "data.yaml").read_text())["names"] == {0: "lane", 1: "sign"}


def test_labelme_conversion_sanitizes_paths_and_exports_masks_as_segments(tmp_path):
    mask = Image.new("L", (20, 20), 0)
    for x in range(5, 15):
        for y in range(5, 15):
            mask.putpixel((x, y), 255)
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")

    (tmp_path / "ann.json").write_text(
        json.dumps(
            {
                "imagePath": "../escape.jpg",
                "imageHeight": 20,
                "imageWidth": 20,
                "shapes": [
                    {
                        "label": "food",
                        "shape_type": "mask",
                        "points": [[0, 0], [20, 20]],
                        "mask": base64.b64encode(buffer.getvalue()).decode(),
                    }
                ],
            }
        )
    )

    save_dir = convert_labelme_json(tmp_path, use_segments=True, save_dir=tmp_path / "out")

    label = (save_dir / "labels" / "escape.txt").read_text().strip().split()
    assert len(label) > 5
    assert not (tmp_path / "escape.txt").exists()


def test_load_labelbox_ndjson(tmp_path):
    file = tmp_path / "labelbox.json"
    file.write_text('{"External ID": "a.jpg"}\n{"External ID": "b.jpg"}\n')

    assert load_labelbox_json(file) == [{"External ID": "a.jpg"}, {"External ID": "b.jpg"}]


def test_vott_conversion_uses_path_outputs(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (200, 100)).save(images / "image1.jpg")
    json_file = tmp_path / "vott.json"
    json_file.write_text(
        json.dumps(
            {
                "asset": {"name": "image1"},
                "regions": [{"tags": ["cat"], "boundingBox": {"left": 10, "top": 20, "width": 40, "height": 30}}],
            }
        )
    )

    save_dir = tmp_path / "out"
    convert_vott_json("data", str(tmp_path / "*.json"), f"{images}/", save_dir=save_dir)

    assert (save_dir / "labels" / "image1.txt").read_text().strip() == "0 0.150000 0.350000 0.200000 0.300000"
