# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import ExifTags
from tqdm import tqdm

# Parameters
img_formats = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng"]  # acceptable image suffixes
vid_formats = ["mov", "avi", "mp4", "mpg", "mpeg", "m4v", "wmv", "mkv"]  # acceptable video suffixes

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def exif_size(img):
    """Returns the EXIF-corrected PIL image size as a tuple (width, height)."""
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270
            s = (s[1], s[0])
    except Exception:
        pass

    return s


def split_rows_simple(file="../data/sm4/out.txt"):  # from utils import *; split_rows_simple()
    """Splits a text file into train, test, and val files based on specified ratios; expects a file path as input."""
    with open(file) as f:
        lines = f.readlines()

    s = Path(file).suffix
    lines = sorted(list(filter(lambda x: len(x) > 0, lines)))
    i, j, k = split_indices(lines, train=0.9, test=0.1, validate=0.0)
    for k, v in {"train": i, "test": j, "val": k}.items():  # key, value pairs
        if v.any():
            new_file = file.replace(s, f"_{k}{s}")
            with open(new_file, "w") as f:
                f.writelines([lines[i] for i in v])


def split_files(out_path, file_name, prefix_path=""):  # split training data
    """Splits file names into separate train, test, and val datasets and writes them to prefixed paths."""
    file_name = list(filter(lambda x: len(x) > 0, file_name))
    file_name = sorted(file_name)
    i, j, k = split_indices(file_name, train=0.9, test=0.1, validate=0.0)
    datasets = {"train": i, "test": j, "val": k}
    for key, item in datasets.items():
        if item.any():
            with open(f"{out_path}_{key}.txt", "a") as file:
                for i in item:
                    file.write(f"{prefix_path}{file_name[i]}\n")


def split_indices(x, train=0.9, test=0.1, validate=0.0, shuffle=True):  # split training data
    """Splits array indices for train, test, and validate datasets according to specified ratios."""
    n = len(x)
    v = np.arange(n)
    if shuffle:
        np.random.shuffle(v)

    i = round(n * train)  # train
    j = round(n * test) + i  # test
    k = round(n * validate) + j  # validate
    return v[:i], v[i:j], v[j:k]  # return indices


def make_dirs(dir="new_dir/"):
    """Creates a directory with subdirectories 'labels' and 'images', removing existing ones."""
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
    for p in dir, dir / "labels", dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir


def write_data_data(fname="data.data", nc=80):
    """Writes a Darknet-style .data file with dataset and training configuration."""
    lines = [
        f"classes = {nc:g}\n",
        "train =../out/data_train.txt\n",
        "valid =../out/data_test.txt\n",
        "names =../out/data.names\n",
        "backup = backup/\n",
        "eval = coco\n",
    ]

    with open(fname, "a") as f:
        f.writelines(lines)


def image_folder2file(folder="images/"):  # from utils import *; image_folder2file()
    """Generates a txt file listing all images in a specified folder; usage: `image_folder2file('path/to/folder/')`."""
    s = glob.glob(f"{folder}*.*")
    with open(f"{folder[:-1]}.txt", "w") as file:
        for l in s:
            file.write(l + "\n")  # write image list


def add_coco_background(path="../data/sm4/", n=1000):  # from utils import *; add_coco_background()
    """
    Adds COCO dataset background images to a specified folder and lists them in outb.txt; usage:

    `add_coco_background('path/', 1000)`.
    """
    p = f"{path}background"
    if os.path.exists(p):
        shutil.rmtree(p)  # delete output folder
    os.makedirs(p)  # make new output folder

    # copy images
    for image in glob.glob("../coco/images/train2014/*.*")[:n]:
        os.system(f"cp {image} {p}")

    # add to outb.txt and make train, test.txt files
    f = f"{path}out.txt"
    fb = f"{path}outb.txt"
    os.system(f"cp {f} {fb}")
    with open(fb, "a") as file:
        file.writelines(i + "\n" for i in glob.glob(f"{p}/*.*"))
    split_rows_simple(file=fb)


def create_single_class_dataset(path="../data/sm3"):  # from utils import *; create_single_class_dataset('../data/sm3/')
    """Creates a single-class version of an existing dataset in the specified path."""
    os.system(f"mkdir {path}_1cls")


def flatten_recursive_folders(path="../../Downloads/data/sm4/"):  # from utils import *; flatten_recursive_folders()
    """Flattens nested folders in 'path/images' and 'path/json' into single 'images_flat' and 'json_flat'
    directories.
    """
    idir, _jdir = f"{path}images/", f"{path}json/"
    nidir, njdir = Path(f"{path}images_flat/"), Path(f"{path}json_flat/")
    n = 0

    # Create output folders
    for p in [nidir, njdir]:
        if os.path.exists(p):
            shutil.rmtree(p)  # delete output folder
        os.makedirs(p)  # make new output folder

    for parent, dirs, files in os.walk(idir):
        for f in tqdm(files, desc=parent):
            f = Path(f)
            stem, suffix = f.stem, f.suffix
            if suffix.lower()[1:] in img_formats:
                n += 1
                stem_new = f"{n:g}_{stem}"
                image_new = nidir / (stem_new + suffix)  # converts all formats to *.jpg
                json_new = njdir / f"{stem_new}.json"

                image = parent / f
                json = Path(parent.replace("images", "json")) / str(f).replace(suffix, ".json")

                os.system(f"cp '{json}' '{json_new}'")
                os.system(f"cp '{image}' '{image_new}'")
                # cv2.imwrite(str(image_new), cv2.imread(str(image)))

    print(f"Flattening complete: {n:g} jsons and images")


def coco91_to_coco80_class():  # converts 80-index (val2014) to 91-index (paper)
    """Converts COCO 91-class index (paper) to 80-class index (2014 challenge)."""
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]
