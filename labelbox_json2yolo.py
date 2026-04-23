# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import os
from pathlib import Path

import requests
import yaml
from PIL import Image
from tqdm import tqdm

from utils import make_dirs


def convert(file, zip=True):
    """Converts Labelbox JSON labels to YOLO format and saves them, with optional zipping."""
    names = []  # class names
    file = Path(file)
    save_dir = make_dirs(file.stem)
    data = load_labelbox_json(file)

    for img in tqdm(data, desc=f"Converting {file}"):
        if not isinstance(img, dict):
            raise TypeError(f"Expected Labelbox record dictionaries, got {type(img).__name__} in {file}")
        im_path = img["Labeled Data"]
        im = Image.open(requests.get(im_path, stream=True).raw if im_path.startswith("http") else im_path)  # open
        width, height = im.size  # image size
        label_path = save_dir / "labels" / Path(img["External ID"]).with_suffix(".txt").name
        image_path = save_dir / "images" / img["External ID"]
        im.save(image_path, quality=95, subsampling=0)

        for label in img["Label"]["objects"]:
            if "bbox" not in label:
                print(f"WARNING: Skipping non-bbox Labelbox object in {img.get('External ID', file.name)}.")
                continue
            # box
            top, left, h, w = label["bbox"].values()  # top, left, height, width
            xywh = [(left + w / 2) / width, (top + h / 2) / height, w / width, h / height]  # xywh normalized

            # class
            cls = label["value"]  # class name
            if cls not in names:
                names.append(cls)

            line = names.index(cls), *xywh  # YOLO format (class_index, xywh)
            with open(label_path, "a") as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")

    # Save dataset.yaml
    d = {
        "path": f"../datasets/{file.stem}  # dataset root dir",
        "train": "images/train  # train images (relative to path) 128 images",
        "val": "images/val  # val images (relative to path) 128 images",
        "test": " # test images (optional)",
        "nc": len(names),
        "names": names,
    }  # dictionary

    with open(save_dir / file.with_suffix(".yaml").name, "w") as f:
        yaml.dump(d, f, sort_keys=False)

    # Zip
    if zip:
        print(f"Zipping as {save_dir}.zip...")
        os.system(f"zip -qr {save_dir}.zip {save_dir}")

    print("Conversion completed successfully!")


def load_labelbox_json(file):
    """Loads Labelbox JSON list, JSON object, or newline-delimited JSON export."""
    text = Path(file).read_text().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return data if isinstance(data, list) else [data]


if __name__ == "__main__":
    convert("export-2021-06-29T15_25_41.934Z.json")
