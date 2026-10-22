# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for utils.py helpers not covered by tests/test_converters.py."""

import io

import numpy as np
import pytest
from PIL import Image

from utils import (
    coco91_to_coco80_class,
    create_single_class_dataset,
    exif_size,
    flatten_recursive_folders,
    image_folder2file,
    img_formats,
    make_dirs,
    split_files,
    split_indices,
    split_rows_simple,
    vid_formats,
    write_data_data,
)


def _jpeg_with_orientation(size, orientation, tmp_path, name="im.jpg"):
    """Write a real JPEG carrying one EXIF orientation tag and reopen it."""
    path = tmp_path / name
    img = Image.new("RGB", size)
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation  # 274 is the EXIF Orientation tag
    img.save(path, format="JPEG", exif=exif)
    return Image.open(path)


# coco91_to_coco80_class -------------------------------------------------------


def test_coco91_to_coco80_map_shape():
    mapping = coco91_to_coco80_class()
    assert len(mapping) == 91
    assert mapping.count(None) == 11  # 91 paper classes collapse to 80
    assert sorted(v for v in mapping if v is not None) == list(range(80))


@pytest.mark.parametrize(
    ("coco91_id", "coco80_id"),
    [
        (1, 0),  # person
        (2, 1),  # bicycle
        (3, 2),  # car
        (13, 11),  # bench: first id after the first dropped slot
        (27, 24),  # backpack
        (46, 40),  # banana, after several dropped slots
        (63, 57),  # laptop (called with category_id - 1 in the converters)
        (90, 79),  # toothbrush, last mapped class
    ],
)
def test_coco91_to_coco80_known_mappings(coco91_id, coco80_id):
    # Converters index the list with category_id - 1.
    assert coco91_to_coco80_class()[coco91_id - 1] == coco80_id


@pytest.mark.parametrize("coco91_id", [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91])
def test_coco91_to_coco80_dropped_ids_are_none(coco91_id):
    assert coco91_to_coco80_class()[coco91_id - 1] is None


# split_indices ----------------------------------------------------------------


def test_split_indices_no_shuffle_exact_partitions():
    train, test, val = split_indices(list(range(10)), train=0.5, test=0.3, validate=0.2, shuffle=False)
    assert train.tolist() == [0, 1, 2, 3, 4]
    assert test.tolist() == [5, 6, 7]
    assert val.tolist() == [8, 9]


def test_split_indices_default_ratio_is_90_10_0():
    train, test, val = split_indices(list(range(10)), shuffle=False)
    assert train.tolist() == list(range(9))
    assert test.tolist() == [9]
    assert val.tolist() == []


def test_split_indices_empty_input():
    train, test, val = split_indices([], shuffle=False)
    assert train.tolist() == test.tolist() == val.tolist() == []


def test_split_indices_rounding_uses_python_round():
    # Boundaries are cumulative round() calls: i = round(5 * 0.5) == 2 and
    # j = round(5 * 0.5) + i == 4 (banker's rounding), so index 4 is in no split.
    train, test, val = split_indices(list(range(5)), train=0.5, test=0.5, validate=0.0, shuffle=False)
    assert train.tolist() == [0, 1]
    assert test.tolist() == [2, 3]
    assert val.tolist() == []


def test_split_indices_all_train():
    train, test, val = split_indices(list(range(4)), train=1.0, test=0.0, validate=0.0, shuffle=False)
    assert train.tolist() == [0, 1, 2, 3]
    assert test.tolist() == val.tolist() == []


def test_split_indices_shuffled_is_seeded_and_complete():
    np.random.seed(42)
    train, test, val = split_indices(list(range(10)), train=0.6, test=0.2, validate=0.2, shuffle=True)
    assert train.tolist() == [8, 1, 5, 0, 7, 2]
    assert test.tolist() == [9, 4]
    assert val.tolist() == [3, 6]
    # Shuffling partitions the index set: no overlap, nothing lost.
    assert sorted(np.concatenate([train, test, val]).tolist()) == list(range(10))


# make_dirs --------------------------------------------------------------------


def test_make_dirs_creates_labels_and_images(tmp_path):
    out = make_dirs(tmp_path / "fresh")
    assert out == tmp_path / "fresh"
    assert (out / "labels").is_dir()
    assert (out / "images").is_dir()


def test_make_dirs_is_destructive_on_rerun(tmp_path):
    out = make_dirs(tmp_path / "out")
    stale = out / "labels" / "stale.txt"
    stale.write_text("old run")
    make_dirs(tmp_path / "out")
    assert not stale.exists()
    assert (out / "labels").is_dir()


# exif_size --------------------------------------------------------------------


def test_exif_size_without_exif_returns_pil_size(tmp_path):
    img = _jpeg_with_orientation((200, 100), None, tmp_path)
    assert exif_size(img) == (200, 100)


@pytest.mark.parametrize("orientation", [6, 8])
def test_exif_size_swaps_dimensions_for_quarter_turns(tmp_path, orientation):
    img = _jpeg_with_orientation((200, 100), orientation, tmp_path, name=f"rot{orientation}.jpg")
    assert exif_size(img) == (100, 200)


@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 7])
def test_exif_size_keeps_dimensions_for_other_orientations(tmp_path, orientation):
    img = _jpeg_with_orientation((200, 100), orientation, tmp_path, name=f"rot{orientation}.jpg")
    assert exif_size(img) == (200, 100)


def test_exif_size_in_memory_image_without_getexif_data():
    # A plain in-memory image has no EXIF dict; size passes through untouched.
    assert exif_size(Image.new("RGB", (37, 11))) == (37, 11)


# write_data_data ---------------------------------------------------------------


def test_write_data_data_exact_content(tmp_path):
    file = tmp_path / "data.data"
    write_data_data(file, nc=3)
    assert file.read_text() == (
        "classes = 3\n"
        "train =../out/data_train.txt\n"
        "valid =../out/data_test.txt\n"
        "names =../out/data.names\n"
        "backup = backup/\n"
        "eval = coco\n"
    )


def test_write_data_data_default_is_80_classes(tmp_path):
    file = tmp_path / "data.data"
    write_data_data(file)
    assert file.read_text().splitlines()[0] == "classes = 80"


def test_write_data_data_appends(tmp_path):
    file = tmp_path / "data.data"
    write_data_data(file, nc=1)
    write_data_data(file, nc=1)
    assert file.read_text().count("classes = 1\n") == 2


# split_files / split_rows_simple ----------------------------------------------


def test_split_files_seeded_exact_output_and_prefix(tmp_path):
    np.random.seed(42)
    names = [f"im{i}.jpg" for i in reversed(range(10))]  # input order must not matter: sorted first
    split_files(str(tmp_path / "data"), names, prefix_path="pre/")
    train = (tmp_path / "data_train.txt").read_text().splitlines()
    test = (tmp_path / "data_test.txt").read_text().splitlines()
    assert train == [f"pre/im{i}.jpg" for i in (8, 1, 5, 0, 7, 2, 9, 4, 3)]
    assert test == ["pre/im6.jpg"]
    assert not (tmp_path / "data_val.txt").exists()  # validate ratio is 0


def test_split_files_empty_and_blank_names_write_nothing(tmp_path):
    split_files(str(tmp_path / "data"), [])
    split_files(str(tmp_path / "data"), ["", ""])
    assert list(tmp_path.iterdir()) == []


def test_split_rows_simple_seeded_exact_output(tmp_path):
    src = tmp_path / "list.txt"
    src.write_text("".join(f"im{i}.jpg\n" for i in reversed(range(10))))
    np.random.seed(42)
    split_rows_simple(str(src))
    assert (tmp_path / "list_train.txt").read_text().splitlines() == [f"im{i}.jpg" for i in (8, 1, 5, 0, 7, 2, 9, 4, 3)]
    assert (tmp_path / "list_test.txt").read_text().splitlines() == ["im6.jpg"]
    assert not (tmp_path / "list_val.txt").exists()


# image_folder2file / format tables / misc --------------------------------------


def test_image_folder2file_lists_every_file(tmp_path):
    folder = tmp_path / "imgs"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(b"x")
    (folder / "b.png").write_bytes(b"x")
    image_folder2file(f"{folder}/")
    listed = (tmp_path / "imgs.txt").read_text().splitlines()
    assert sorted(listed) == sorted([str(folder / "a.jpg"), str(folder / "b.png")])


def test_format_tables_are_exact_and_disjoint():
    assert img_formats == ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng"]
    assert vid_formats == ["mov", "avi", "mp4", "mpg", "mpeg", "m4v", "wmv", "mkv"]
    assert not set(img_formats) & set(vid_formats)


def test_create_single_class_dataset_makes_sibling_dir(tmp_path):
    base = tmp_path / "ds"
    create_single_class_dataset(str(base))
    assert (tmp_path / "ds_1cls").is_dir()


def test_flatten_recursive_folders_empty_tree(tmp_path, capsys):
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "json").mkdir()
    flatten_recursive_folders(f"{data}/")
    assert (data / "images_flat").is_dir()
    assert (data / "json_flat").is_dir()
    assert "Flattening complete: 0" in capsys.readouterr().out


def test_exif_size_from_bytesio_roundtrip():
    buffer = io.BytesIO()
    Image.new("RGB", (640, 480)).save(buffer, format="PNG")
    buffer.seek(0)
    assert exif_size(Image.open(buffer)) == (640, 480)
