# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for general_json2yolo.py helpers and converters.

Covers edges tests/test_converters.py does not: coordinate boundaries,
class mapping, empty/degenerate annotations, polygon handling, output
precision, invalid inputs, and the legacy INFOLKS / VoTT / ATH converters.
"""

import base64
import io
import json

import numpy as np
import pytest
import yaml
from PIL import Image

from general_json2yolo import (
    bbox_from_keypoints,
    coco_keypoints,
    coco_names,
    convert_ath_json,
    convert_coco_json,
    convert_infolks_json,
    convert_labelme_json,
    convert_vott_json,
    labelme_image_file,
    labelme_points,
    mask2points,
    merge_multi_segment,
    min_index,
    parse_args,
    rle2polygon,
    safe_relative_path,
    write_coco_yaml,
    write_dataset_yaml,
    yolo_bbox,
    yolo_segment,
)
from utils import coco91_to_coco80_class


def _write_coco(tmp_path, data, name="instances_train.json"):
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir(exist_ok=True)
    (ann_dir / name).write_text(json.dumps(data))
    return ann_dir


def _coco_doc(annotations, categories=None, width=200, height=100, file_name="im.jpg"):
    return {
        "images": [{"id": 1, "height": height, "width": width, "file_name": file_name}],
        "categories": categories if categories is not None else [{"id": 1, "name": "person"}],
        "annotations": annotations,
    }


# yolo_bbox / yolo_segment ------------------------------------------------------


def test_yolo_bbox_basic_rectangle():
    # Corners (10, 20)-(50, 60) on a 200x100 image.
    assert yolo_bbox([[10, 20], [50, 60]], 200, 100) == [0.15, 0.4, 0.2, 0.4]


def test_yolo_bbox_full_image_and_origin():
    assert yolo_bbox([[0, 0], [200, 100]], 200, 100) == [0.5, 0.5, 1.0, 1.0]
    assert yolo_bbox([[0, 0], [100, 50]], 200, 100) == [0.25, 0.25, 0.5, 0.5]


def test_yolo_bbox_corner_order_does_not_matter():
    assert yolo_bbox([[50, 60], [10, 20]], 200, 100) == yolo_bbox([[10, 20], [50, 60]], 200, 100)


def test_yolo_bbox_uses_extremes_of_all_points():
    # A polygon's bbox spans min/max of every vertex, not just first/last.
    assert yolo_bbox([[30, 10], [90, 90], [10, 50]], 200, 100) == [0.25, 0.5, 0.4, 0.8]


@pytest.mark.parametrize(
    "points",
    [
        [[10, 20]],  # single point: zero width and height
        [[10, 20], [10, 60]],  # zero width
        [[10, 20], [50, 20]],  # zero height
    ],
)
def test_yolo_bbox_degenerate_returns_empty(points):
    assert yolo_bbox(points, 200, 100) == []


def test_yolo_segment_normalizes_each_vertex():
    assert yolo_segment([[0, 0], [100, 50], [200, 100]], 200, 100) == [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]


@pytest.mark.parametrize("points", [[[0, 0]], [[0, 0], [100, 50]]])
def test_yolo_segment_needs_at_least_three_points(points):
    assert yolo_segment(points, 200, 100) == []


# keypoint helpers ---------------------------------------------------------------


def test_bbox_from_keypoints_uses_visible_only():
    # Third keypoint has visibility 0 and must not stretch the box.
    ann = {"keypoints": [10, 20, 2, 30, 60, 1, 999, 999, 0]}
    assert bbox_from_keypoints(ann) == [10.0, 20.0, 20.0, 40.0]


@pytest.mark.parametrize(
    "ann",
    [
        {"keypoints": [10, 20, 0, 30, 60, 0]},  # nothing visible
        {"keypoints": []},
        {},  # key missing entirely
    ],
)
def test_bbox_from_keypoints_without_visible_points_is_zero_box(ann):
    assert bbox_from_keypoints(ann) == [0, 0, 0, 0]


def test_coco_keypoints_normalizes_xy_but_not_visibility():
    ann = {"keypoints": [20, 30, 2, 40, 50, 0]}
    assert coco_keypoints(ann, 200, 100) == [0.1, 0.3, 2.0, 0.2, 0.5, 0.0]


def test_coco_keypoints_empty():
    assert coco_keypoints({}, 200, 100) == []


# min_index / merge_multi_segment -------------------------------------------------


def test_min_index_finds_closest_pair():
    assert min_index(np.array([[0.0, 0.0], [10.0, 10.0]]), np.array([[9.0, 9.0], [100.0, 100.0]])) == (1, 0)
    assert min_index(np.array([[5.0, 5.0]]), np.array([[5.0, 6.0], [0.0, 0.0]])) == (0, 0)


def test_merge_multi_segment_stitches_two_segments():
    merged = merge_multi_segment([[0, 0, 10, 0, 10, 10], [100, 100, 110, 100]])
    assert [s.tolist() for s in merged] == [
        [[10, 10], [0, 0], [10, 0], [10, 10]],
        [[100, 100], [110, 100], [100, 100]],
    ]


# LabelMe path / point helpers ------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a/b.jpg", "a/b.jpg"),
        ("a\\b.jpg", "a/b.jpg"),  # Windows separators are normalized
        ("./a.jpg", "a.jpg"),
        ("../evil.jpg", "evil.jpg"),  # traversal collapsed to the file name
        ("/abs/x.jpg", "x.jpg"),  # absolute paths collapsed to the file name
        ("", "fallback.jpg"),
        (None, "fallback.jpg"),
    ],
)
def test_safe_relative_path(raw, expected):
    from pathlib import Path

    assert safe_relative_path(raw, "fallback.jpg") == Path(expected)


def test_labelme_image_file_relative_and_absolute(tmp_path):
    json_file = tmp_path / "sub" / "ann.json"
    assert labelme_image_file(json_file, "pic.jpg") == tmp_path / "sub" / "pic.jpg"
    assert labelme_image_file(json_file, "/elsewhere/pic.jpg").is_absolute()


def test_labelme_points_circle_becomes_bounding_square():
    # Center (50, 50), edge point 10 px away -> square from (40, 40) to (60, 60).
    shape = {"shape_type": "circle", "points": [[50, 50], [60, 50]]}
    points = labelme_points(shape, 200, 100)
    assert [[float(v) for v in p] for p in points] == [[40.0, 40.0], [60.0, 60.0]]


def test_labelme_points_non_special_shapes_pass_through():
    shape = {"shape_type": "rectangle", "points": [[1, 2], [3, 4]]}
    assert labelme_points(shape, 200, 100) == [[1, 2], [3, 4]]


def _png_b64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_mask2points_square_mask():
    mask = Image.new("L", (20, 20), 0)
    for x in range(5, 15):
        for y in range(5, 15):
            mask.putpixel((x, y), 255)
    assert mask2points(_png_b64(mask), [[0, 0], [20, 20]], 20, 20) == [[5, 5], [5, 14], [14, 14], [14, 5]]


def test_mask2points_blank_mask_returns_empty():
    assert mask2points(_png_b64(Image.new("L", (20, 20), 0)), [[0, 0]], 20, 20) == []


# COCO names / YAML -------------------------------------------------------------------


def test_coco_names_without_91to80_mapping():
    data = {"categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]}
    assert coco_names(data, coco91_to_coco80_class(), False) == {0: "person", 2: "car"}


def test_coco_names_with_91to80_mapping_and_dropped_class():
    data = {"categories": [{"id": 13, "name": "bench"}, {"id": 12, "name": "dropped"}]}
    # 91-id 13 maps to 80-id 11; 91-id 12 maps to None and is omitted.
    assert coco_names(data, coco91_to_coco80_class(), True) == {11: "bench"}


def test_coco_names_empty_categories():
    assert coco_names({}, coco91_to_coco80_class(), False) == {}
    assert coco_names({"categories": []}, coco91_to_coco80_class(), False) == {}


def test_write_coco_yaml_skips_file_when_no_names(tmp_path):
    file = tmp_path / "x.yaml"
    write_coco_yaml(file, {"categories": []}, coco91_to_coco80_class(), False)
    assert not file.exists()


def test_write_dataset_yaml_accepts_dict_names(tmp_path):
    file = tmp_path / "data.yaml"
    write_dataset_yaml(file, {2: "b", 0: "a"})
    loaded = yaml.safe_load(file.read_text())
    assert loaded["names"] == {0: "a", 2: "b"}
    assert loaded["train"] == loaded["val"] == "images"
    assert loaded["path"] == str(tmp_path)


def test_rle2polygon_uncompressed_rle():
    # 10x10 mask, first 90 pixels background, last column foreground.
    assert rle2polygon({"size": [10, 10], "counts": [90, 10]}) == [[9, 0, 9, 9]]


# parse_args ---------------------------------------------------------------------------


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    args = parse_args()
    assert args.source == "COCO"
    assert args.save_dir == "new_dir"
    assert args.use_segments is False and args.use_keypoints is False and args.cls91to80 is False


def test_parse_args_explicit_values(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--source", "LabelMe", "--json-dir", "/tmp/x", "--use-segments", "--cls91to80", "--name", "stem"],
    )
    args = parse_args()
    assert (args.source, args.json_dir, args.use_segments, args.cls91to80, args.name) == (
        "LabelMe",
        "/tmp/x",
        True,
        True,
        "stem",
    )


def test_parse_args_rejects_unknown_source(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--source", "Nope"])
    with pytest.raises(SystemExit):
        parse_args()


# COCO end-to-end edges -------------------------------------------------------------------


def test_coco_skips_crowd_missing_category_zero_size_and_duplicates(tmp_path):
    ann_dir = _write_coco(
        tmp_path,
        _coco_doc(
            [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 200, 100]},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 200, 100]},
                {"id": 3, "image_id": 1, "category_id": 1, "bbox": [0, 0, 200, 100]},  # exact duplicate
                {"id": 4, "image_id": 1, "category_id": 1, "bbox": [10, 10, 0, 20]},  # zero width
                {"id": 5, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 0]},  # zero height
                {"id": 6, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 1},
                {"id": 7, "image_id": 1, "bbox": [10, 10, 20, 20]},  # no category_id
            ],
            categories=[{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
        ),
    )
    save_dir = convert_coco_json(ann_dir, save_dir=tmp_path / "out")
    lines = (save_dir / "labels" / "train" / "im.txt").read_text().splitlines()
    assert lines == ["0 0.5 0.5 1 1", "1 0.5 0.5 1 1"]


def test_coco_cls91to80_conversion_and_yaml(tmp_path):
    ann_dir = _write_coco(
        tmp_path,
        _coco_doc(
            [
                {"id": 1, "image_id": 1, "category_id": 13, "bbox": [10, 10, 20, 20]},
                {"id": 2, "image_id": 1, "category_id": 12, "bbox": [30, 30, 20, 20]},  # maps to None
            ],
            categories=[{"id": 13, "name": "bench"}, {"id": 12, "name": "dropped"}],
            width=100,
            height=100,
        ),
    )
    save_dir = convert_coco_json(ann_dir, cls91to80=True, save_dir=tmp_path / "out")
    assert (save_dir / "labels" / "train" / "im.txt").read_text().strip() == "11 0.2 0.2 0.2 0.2"
    assert yaml.safe_load((save_dir / "instances_train.yaml").read_text()) == {"names": {11: "bench"}}


def test_coco_output_precision_uses_g_format(tmp_path):
    doc = _coco_doc(
        [{"id": 1, "image_id": 7, "category_id": 1, "bbox": [1, 1, 1, 1]}],
        width=333,
        height=333,
    )
    doc["images"] = [{"id": 7, "height": 333, "width": 333, "file_name": "im.jpg"}]
    ann_dir = _write_coco(tmp_path, doc)
    save_dir = convert_coco_json(ann_dir, save_dir=tmp_path / "out")
    # Center 1.5/333 and size 1/333, printed with %g (6 significant digits).
    assert (save_dir / "labels" / "train" / "im.txt").read_text().strip() == "0 0.0045045 0.0045045 0.003003 0.003003"


def test_coco_empty_annotations_writes_yaml_but_no_labels(tmp_path):
    ann_dir = _write_coco(tmp_path, _coco_doc([]))
    save_dir = convert_coco_json(ann_dir, save_dir=tmp_path / "out")
    assert yaml.safe_load((save_dir / "instances_train.yaml").read_text()) == {"names": {0: "person"}}
    assert list((save_dir / "labels").rglob("*.txt")) == []


def test_coco_missing_top_level_keys_are_skipped(tmp_path, capsys):
    ann_dir = _write_coco(tmp_path, {"images": []}, name="bad.json")
    save_dir = convert_coco_json(ann_dir, save_dir=tmp_path / "out")
    assert "expected COCO keys" in capsys.readouterr().out
    assert list(save_dir.rglob("*.txt")) == []


def test_coco_no_json_files_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No JSON files found"):
        convert_coco_json(empty, save_dir=tmp_path / "out")


def test_coco_segments_fall_back_to_bbox_without_segmentation(tmp_path):
    ann_dir = _write_coco(
        tmp_path, _coco_doc([{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 40, 30]}])
    )
    save_dir = convert_coco_json(ann_dir, use_segments=True, save_dir=tmp_path / "out")
    assert (save_dir / "labels" / "train" / "im.txt").read_text().strip() == "0 0.15 0.35 0.2 0.3"


def test_coco_multi_segment_annotations_are_stitched(tmp_path):
    ann_dir = _write_coco(
        tmp_path,
        _coco_doc(
            [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 20, 20],
                    "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10], [20, 20, 30, 20, 30, 30]],
                }
            ],
            width=100,
            height=100,
        ),
    )
    save_dir = convert_coco_json(ann_dir, use_segments=True, save_dir=tmp_path / "out")
    assert (save_dir / "labels" / "train" / "im.txt").read_text().strip() == (
        "0 0.1 0.1 0 0.1 0 0 0.1 0 0.1 0.1 0.2 0.2 0.3 0.2 0.3 0.3 0.2 0.2"
    )


def test_coco_rle_segmentation_end_to_end(tmp_path):
    ann_dir = _write_coco(
        tmp_path,
        _coco_doc(
            [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [9, 0, 1, 10],
                    "segmentation": {"size": [10, 10], "counts": [90, 10]},
                }
            ],
            width=10,
            height=10,
        ),
    )
    save_dir = convert_coco_json(ann_dir, use_segments=True, save_dir=tmp_path / "out")
    assert (save_dir / "labels" / "train" / "im.txt").read_text().strip() == "0 0.9 0 0.9 0.9"


def test_coco_annotation_for_unknown_image_raises_keyerror(tmp_path):
    ann_dir = _write_coco(
        tmp_path,
        {
            "images": [{"id": 1, "height": 10, "width": 10, "file_name": "im.jpg"}],
            "categories": [],
            "annotations": [{"id": 1, "image_id": 99, "category_id": 1, "bbox": [0, 0, 5, 5]}],
        },
    )
    with pytest.raises(KeyError):
        convert_coco_json(ann_dir, save_dir=tmp_path / "out")


def test_coco_derives_bbox_from_visible_keypoints_when_bbox_missing(tmp_path):
    ann_dir = _write_coco(
        tmp_path,
        _coco_doc(
            [{"id": 1, "image_id": 1, "category_id": 1, "keypoints": [10, 20, 2, 30, 60, 2, 0, 0, 0]}],
        ),
    )
    save_dir = convert_coco_json(ann_dir, save_dir=tmp_path / "out")
    # Visible points span (10, 20)-(30, 60): center (0.1, 0.4), size (0.1, 0.4).
    assert (save_dir / "labels" / "train" / "im.txt").read_text().strip() == "0 0.1 0.4 0.1 0.4"


# LabelMe end-to-end edges ------------------------------------------------------------------


def test_labelme_no_json_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No JSON files found"):
        convert_labelme_json(tmp_path, save_dir=tmp_path / "out")


def test_labelme_missing_dimensions_are_skipped(tmp_path, capsys):
    (tmp_path / "a.json").write_text(json.dumps({"imagePath": "a.jpg", "shapes": [{"label": "x", "points": [[0, 0], [1, 1]]}]}))
    save_dir = convert_labelme_json(tmp_path, save_dir=tmp_path / "out")
    assert "missing imageWidth/imageHeight" in capsys.readouterr().out
    assert list((save_dir / "labels").rglob("*.txt")) == []
    assert yaml.safe_load((save_dir / "data.yaml").read_text())["names"] == {}


def test_labelme_empty_shapes_produce_no_label_file(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps({"imagePath": "a.jpg", "imageHeight": 100, "imageWidth": 200, "shapes": []})
    )
    save_dir = convert_labelme_json(tmp_path, save_dir=tmp_path / "out")
    assert list((save_dir / "labels").rglob("*.txt")) == []
    assert yaml.safe_load((save_dir / "data.yaml").read_text())["names"] == {}


def test_labelme_circle_and_degenerate_shapes(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps(
            {
                "imagePath": "a.jpg",
                "imageHeight": 100,
                "imageWidth": 100,
                "shapes": [
                    {"label": "circ", "shape_type": "circle", "points": [[50, 50], [60, 50]]},
                    {"label": "pt", "shape_type": "polygon", "points": [[10, 10]]},
                    {"label": "flat", "shape_type": "polygon", "points": [[10, 10], [10, 50]]},
                ],
            }
        )
    )
    save_dir = convert_labelme_json(tmp_path, use_segments=False, save_dir=tmp_path / "out")
    # Only the circle survives as a box; degenerate shapes are dropped...
    assert (save_dir / "labels" / "a.txt").read_text().strip() == "0 0.5 0.5 0.2 0.2"
    # ...but their labels were still registered in first-seen order.
    assert yaml.safe_load((save_dir / "data.yaml").read_text())["names"] == {0: "circ", 1: "pt", 2: "flat"}


def test_labelme_two_point_polygon_falls_back_to_bbox_in_segment_mode(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps(
            {
                "imagePath": "a.jpg",
                "imageHeight": 100,
                "imageWidth": 200,
                "shapes": [{"label": "seg2", "shape_type": "polygon", "points": [[0, 0], [100, 100]]}],
            }
        )
    )
    save_dir = convert_labelme_json(tmp_path, use_segments=True, save_dir=tmp_path / "out")
    assert (save_dir / "labels" / "a.txt").read_text().strip() == "0 0.25 0.5 0.5 1"


def test_labelme_default_label_and_first_seen_class_order(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps(
            {
                "imagePath": "a.jpg",
                "imageHeight": 100,
                "imageWidth": 100,
                "shapes": [
                    {"shape_type": "rectangle", "points": [[0, 0], [50, 50]]},  # no label -> "object"
                    {"label": "zebra", "shape_type": "rectangle", "points": [[50, 50], [100, 100]]},
                    {"label": "zebra", "shape_type": "rectangle", "points": [[0, 50], [50, 100]]},
                ],
            }
        )
    )
    save_dir = convert_labelme_json(tmp_path, use_segments=False, save_dir=tmp_path / "out")
    lines = (save_dir / "labels" / "a.txt").read_text().strip().splitlines()
    assert lines == ["0 0.25 0.25 0.5 0.5", "1 0.75 0.75 0.5 0.5", "1 0.25 0.75 0.5 0.5"]
    assert yaml.safe_load((save_dir / "data.yaml").read_text())["names"] == {0: "object", 1: "zebra"}


# VoTT / INFOLKS / ATH ------------------------------------------------------------------------


def test_vott_sorted_names_zero_width_dropped_and_missing_image_warned(tmp_path, capsys):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (100, 100)).save(images / "a.jpg")
    (tmp_path / "v1.json").write_text(
        json.dumps(
            {
                "asset": {"name": "a"},
                "regions": [
                    {"tags": ["zebra"], "boundingBox": {"left": 0, "top": 0, "width": 100, "height": 50}},
                    {"tags": ["apple"], "boundingBox": {"left": 50, "top": 50, "width": 0, "height": 10}},
                ],
            }
        )
    )
    (tmp_path / "v2.json").write_text(
        json.dumps(
            {
                "asset": {"name": "missing"},
                "regions": [{"tags": ["apple"], "boundingBox": {"left": 0, "top": 0, "width": 10, "height": 10}}],
            }
        )
    )
    save_dir = tmp_path / "out"
    convert_vott_json("data", str(tmp_path / "*.json"), f"{images}/", save_dir=save_dir)
    assert (save_dir / "data.names").read_text() == "apple\nzebra\n"  # sorted, not first-seen
    # zebra is class 1 after sorting; the zero-width apple region is dropped.
    assert (save_dir / "labels" / "a.txt").read_text().strip() == "1 0.500000 0.250000 1.000000 0.500000"
    assert "missing" in capsys.readouterr().out


def test_infolks_lowercases_sorts_classes_and_writes_data_file(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (200, 100)).save(images / "im1.jpg")
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    (json_dir / "im1.json").write_text(
        json.dumps(
            {
                "output": {
                    "objects": [
                        {"classTitle": "Cat", "points": {"exterior": [[10, 20], [50, 60]]}},
                        {"classTitle": "dog", "points": {"exterior": [[0, 0], [200, 100]]}},
                        {"classTitle": "cat", "points": {"exterior": [[5, 5], [5, 5]]}},  # zero area, dropped
                    ]
                }
            }
        )
    )
    save_dir = tmp_path / "out"
    convert_infolks_json("data", str(json_dir / "*.json"), f"{images}/", save_dir=save_dir)
    assert (save_dir / "data.names").read_text() == "cat\ndog\n"
    assert (save_dir / "labels" / "im1.txt").read_text().splitlines() == [
        "0 0.150000 0.400000 0.200000 0.400000",
        "1 0.500000 0.500000 1.000000 1.000000",
    ]
    assert (save_dir / "data.data").read_text().splitlines()[0] == "classes = 2"
    assert (save_dir / "data.txt").read_text().strip() == str(images / "im1.jpg")


def test_ath_single_class_conversion(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (200, 100)).save(src / "im1.jpg")
    (src / "ann.json").write_text(
        json.dumps(
            {
                "_via_img_metadata": {
                    "k1": {
                        "filename": "im1.jpg",
                        "regions": [{"shape_attributes": {"x": 10, "y": 20, "width": 40, "height": 30}}],
                    }
                }
            }
        )
    )
    save_dir = tmp_path / "out"
    convert_ath_json(str(src), save_dir=save_dir)
    assert (save_dir / "labels" / "im1.txt").read_text().strip() == "0 0.150000 0.350000 0.200000 0.300000"
    assert (save_dir / "data.names").read_text() == "knife\n"
    assert (save_dir / "data.data").read_text().splitlines()[0] == "classes = 1"
    assert (save_dir / "images" / "im1.jpg").exists()
