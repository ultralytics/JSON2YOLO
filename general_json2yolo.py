# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import base64
import contextlib
import io
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import yaml
from PIL import Image

from utils import *


# Convert INFOLKS JSON file into YOLO-format labels ----------------------------
def convert_infolks_json(name, files, img_path, save_dir="new_dir"):
    """Converts INFOLKS JSON annotations to YOLO-format labels."""
    path = make_dirs(save_dir)

    # Import json
    data = []
    for file in glob.glob(files):
        with open(file) as f:
            jdata = json.load(f)
            jdata["json_file"] = file
            data.append(jdata)

    # Write images and shapes
    name = path / name
    _file_id, file_name, wh, cat = [], [], [], []
    for x in tqdm(data, desc="Files and Shapes"):
        f = glob.glob(img_path + Path(x["json_file"]).stem + ".*")[0]
        file_name.append(f)
        wh.append(exif_size(Image.open(f)))  # (width, height)
        cat.extend(a["classTitle"].lower() for a in x["output"]["objects"])  # categories

        # filename
        with open(name.with_suffix(".txt"), "a") as file:
            file.write(f"{f}\n")

    # Write *.names file
    names = sorted(np.unique(cat))
    # names.pop(names.index('Missing product'))  # remove
    with open(name.with_suffix(".names"), "a") as file:
        [file.write(f"{a}\n") for a in names]

    # Write labels file
    for i, x in enumerate(tqdm(data, desc="Annotations")):
        label_name = Path(file_name[i]).stem + ".txt"

        with open(path / "labels" / label_name, "a") as file:
            for a in x["output"]["objects"]:
                # if a['classTitle'] == 'Missing product':
                #    continue  # skip

                category_id = names.index(a["classTitle"].lower())

                # The INFOLKS bounding box format is [x-min, y-min, x-max, y-max]
                box = np.array(a["points"]["exterior"], dtype=np.float32).ravel()
                box[[0, 2]] /= wh[i][0]  # normalize x by width
                box[[1, 3]] /= wh[i][1]  # normalize y by height
                box = [box[[0, 2]].mean(), box[[1, 3]].mean(), box[2] - box[0], box[3] - box[1]]  # xywh
                if (box[2] > 0.0) and (box[3] > 0.0):  # if w > 0 and h > 0
                    file.write("{:g} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(category_id, *box))

    # Split data into train, test, and validate files
    split_files(name, file_name)
    write_data_data(name.with_suffix(".data"), nc=len(names))
    print(f"Done. Output saved to {path.absolute()}")


# Convert vott JSON file into YOLO-format labels -------------------------------
def convert_vott_json(name, files, img_path, save_dir="new_dir"):
    """Converts VoTT JSON files to YOLO-format labels and organizes dataset structure."""
    path = make_dirs(save_dir)
    name = path / name

    # Import json
    data = []
    for file in glob.glob(files):
        with open(file) as f:
            jdata = json.load(f)
            jdata["json_file"] = file
            data.append(jdata)

    # Get all categories
    file_name, wh, cat = [], [], []
    for i, x in enumerate(tqdm(data, desc="Files and Shapes")):
        with contextlib.suppress(Exception):
            cat.extend(a["tags"][0] for a in x["regions"])  # categories

    # Write *.names file
    names = sorted(set(cat))
    with open(name.with_suffix(".names"), "a") as file:
        [file.write(f"{a}\n") for a in names]

    # Write labels file
    n1, n2 = 0, 0
    missing_images = []
    for i, x in enumerate(tqdm(data, desc="Annotations")):
        f = glob.glob(img_path + x["asset"]["name"] + ".jpg")
        if len(f):
            f = f[0]
            file_name.append(f)
            wh = exif_size(Image.open(f))  # (width, height)

            n1 += 1
            if (len(f) > 0) and (wh[0] > 0) and (wh[1] > 0):
                n2 += 1

                # append filename to list
                with open(name.with_suffix(".txt"), "a") as file:
                    file.write(f"{f}\n")

                # write labelsfile
                label_name = Path(f).stem + ".txt"
                with open(path / "labels" / label_name, "a") as file:
                    for a in x["regions"]:
                        category_id = names.index(a["tags"][0])

                        # The INFOLKS bounding box format is [x-min, y-min, x-max, y-max]
                        box = a["boundingBox"]
                        box = np.array([box["left"], box["top"], box["width"], box["height"]], dtype=np.float32).ravel()
                        box[[0, 2]] /= wh[0]  # normalize x by width
                        box[[1, 3]] /= wh[1]  # normalize y by height
                        box = [box[0] + box[2] / 2, box[1] + box[3] / 2, box[2], box[3]]  # xywh

                        if (box[2] > 0.0) and (box[3] > 0.0):  # if w > 0 and h > 0
                            file.write("{:g} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(category_id, *box))
        else:
            missing_images.append(x["asset"]["name"])

    print(f"Attempted {i:g} json imports, found {n1:g} images, imported {n2:g} annotations successfully")
    if len(missing_images):
        print("WARNING, missing images:", missing_images)

    # Split data into train, test, and validate files
    split_files(name, file_name)
    print(f"Done. Output saved to {path.absolute()}")


# Convert ath JSON file into YOLO-format labels --------------------------------
def convert_ath_json(json_dir, save_dir="new_dir"):  # dir contains json annotations and images
    """Converts ath JSON annotations to YOLO-format labels, resizes images, and organizes data for training."""
    dir = make_dirs(save_dir)  # output directory

    jsons = []
    for dirpath, dirnames, filenames in os.walk(json_dir):
        jsons.extend(
            os.path.join(dirpath, filename) for filename in [f for f in filenames if f.lower().endswith(".json")]
        )

    # Import json
    n1, n2, n3 = 0, 0, 0
    missing_images, file_name = [], []
    for json_file in sorted(jsons):
        with open(json_file) as f:
            data = json.load(f)

        # # Get classes
        # try:
        #     classes = list(data['_via_attributes']['region']['class']['options'].values())  # classes
        # except:
        #     classes = list(data['_via_attributes']['region']['Class']['options'].values())  # classes

        # # Write *.names file
        # names = pd.unique(classes)  # preserves sort order
        # with open(dir + 'data.names', 'w') as f:
        #     [f.write('%s\n' % a) for a in names]

        # Write labels file
        for x in tqdm(data["_via_img_metadata"].values(), desc=f"Processing {json_file}"):
            image_file = str(Path(json_file).parent / x["filename"])
            f = glob.glob(image_file)  # image file
            if len(f):
                f = f[0]
                file_name.append(f)
                wh = exif_size(Image.open(f))  # (width, height)

                n1 += 1  # all images
                if len(f) > 0 and wh[0] > 0 and wh[1] > 0:
                    label_file = dir / "labels" / f"{Path(f).stem}.txt"

                    nlabels = 0
                    try:
                        with open(label_file, "a") as file:  # write labelsfile
                            # try:
                            #     category_id = int(a['region_attributes']['class'])
                            # except:
                            #     category_id = int(a['region_attributes']['Class'])
                            category_id = 0  # single-class

                            for a in x["regions"]:
                                # bounding box format is [x-min, y-min, x-max, y-max]
                                box = a["shape_attributes"]
                                box = np.array(
                                    [box["x"], box["y"], box["width"], box["height"]], dtype=np.float32
                                ).ravel()
                                box[[0, 2]] /= wh[0]  # normalize x by width
                                box[[1, 3]] /= wh[1]  # normalize y by height
                                box = [
                                    box[0] + box[2] / 2,
                                    box[1] + box[3] / 2,
                                    box[2],
                                    box[3],
                                ]  # xywh (left-top to center x-y)

                                if box[2] > 0.0 and box[3] > 0.0:  # if w > 0 and h > 0
                                    file.write("{:g} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(category_id, *box))
                                    n3 += 1
                                    nlabels += 1

                        if nlabels == 0:  # remove non-labelled images from dataset
                            label_file.unlink(missing_ok=True)
                            continue  # next file

                        # write image
                        img_size = 4096  # resize to maximum
                        img = cv2.imread(f)  # BGR
                        assert img is not None, "Image Not Found " + f
                        r = img_size / max(img.shape)  # size ratio
                        if r < 1:  # downsize if necessary
                            h, w, _ = img.shape
                            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)

                        ifile = dir / "images" / Path(f).name
                        if cv2.imwrite(str(ifile), img):  # if success append image to list
                            with open(dir / "data.txt", "a") as file:
                                file.write(f"{ifile}\n")
                            n2 += 1  # correct images

                    except Exception:
                        label_file.unlink(missing_ok=True)
                        print(f"problem with {f}")

            else:
                missing_images.append(image_file)

    nm = len(missing_images)  # number missing
    print(
        f"\nFound {len(jsons):g} JSONs with {n3:g} labels over {n1:g} images. Found {n1 - nm:g} images, labeled {n2:g} images successfully"
    )
    if len(missing_images):
        print("WARNING, missing images:", missing_images)

    # Write *.names file
    names = ["knife"]  # preserves sort order
    with open(dir / "data.names", "w") as f:
        [f.write(f"{a}\n") for a in names]

    # Split data into train, test, and validate files
    split_rows_simple(dir / "data.txt")
    write_data_data(dir / "data.data", nc=1)
    print(f"Done. Output saved to {Path(dir).absolute()}")


def convert_coco_json(
    json_dir="../coco/annotations/", use_segments=False, use_keypoints=False, cls91to80=False, save_dir="new_dir"
):
    """Converts COCO JSON format to YOLO label format, with options for segments, keypoints, and class mapping."""
    save_dir = make_dirs(save_dir)  # output directory
    coco80 = coco91_to_coco80_class()
    json_files = sorted(Path(json_dir).resolve().glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {Path(json_dir).resolve()}")

    # Import json
    for json_file in json_files:
        fn = Path(save_dir) / "labels" / json_file.stem.replace("instances_", "")  # folder name
        fn.mkdir(parents=True, exist_ok=True)
        with open(json_file) as f:
            data = json.load(f)
        if not {"images", "annotations"}.issubset(data):
            print(f"WARNING: Skipping {json_file}, expected COCO keys 'images' and 'annotations'.")
            continue
        write_coco_yaml(Path(save_dir) / f"{json_file.stem}.yaml", data, coco80, cls91to80)

        # Create image dict
        images = {"{:g}".format(x["id"]): x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:g}"]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue
                if "category_id" not in ann:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann.get("bbox") or bbox_from_keypoints(ann), dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # class
                if cls is None:
                    continue
                box = [cls, *box.tolist()]
                if box not in bboxes:
                    bboxes.append(box)
                else:
                    continue
                # Segments
                if use_segments:
                    segmentation = ann.get("segmentation", [])
                    if isinstance(segmentation, dict):
                        segmentation = rle2polygon(segmentation)
                    if len(segmentation) > 1:
                        s = merge_multi_segment(segmentation)
                        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                    elif len(segmentation) == 1:
                        s = [j for i in segmentation for j in i]  # all segments concatenated
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                    else:
                        s = []
                    segments.append([cls, *s] if s else [])
                if use_keypoints:
                    keypoints.append(box + coco_keypoints(ann, w, h))

            # Write
            label_file = (fn / Path(f).name if Path(f).is_absolute() else fn / f).with_suffix(".txt")
            label_file.parent.mkdir(parents=True, exist_ok=True)
            with open(label_file, "a") as file:
                for i in range(len(bboxes)):
                    line = keypoints[i] if use_keypoints else segments[i] if use_segments and segments[i] else bboxes[i]
                    line = tuple(line)
                    file.write(("%g " * len(line)).rstrip() % line + "\n")
    return save_dir


def convert_labelme_json(json_dir, use_segments=True, save_dir="new_dir"):
    """Converts LabelMe JSON annotations to YOLO detection or segmentation labels."""
    save_dir = make_dirs(save_dir)
    json_files = sorted(Path(json_dir).resolve().rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {Path(json_dir).resolve()}")

    names = []
    converted = 0
    for json_file in tqdm(json_files, desc="LabelMe"):
        with open(json_file) as f:
            data = json.load(f)
        width, height = data.get("imageWidth"), data.get("imageHeight")
        if not width or not height:
            print(f"WARNING: Skipping {json_file}, missing imageWidth/imageHeight.")
            continue

        image_name = data.get("imagePath") or json_file.with_suffix(".jpg").name
        image_file = labelme_image_file(json_file, image_name)
        output_name = safe_relative_path(image_name, json_file.with_suffix(".jpg").name)
        label_file = (save_dir / "labels" / output_name).with_suffix(".txt")
        label_file.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for shape in data.get("shapes", []):
            label = shape.get("label", "object")
            if label not in names:
                names.append(label)
            cls = names.index(label)
            points = labelme_points(shape, width, height)
            if len(points) < 2:
                continue

            shape_type = shape.get("shape_type", "polygon")
            segment = yolo_segment(points, width, height)
            if use_segments and shape_type in {"polygon", "linestrip", "mask"} and segment:
                line = [cls, *segment]
            else:
                box = yolo_bbox(points, width, height)
                if not box:
                    continue
                line = [cls, *box]
            lines.append(("%g " * len(line)).rstrip() % tuple(line))

        if lines:
            label_file.write_text("\n".join(lines) + "\n")
            converted += 1

        if image_file.exists():
            output_image = save_dir / "images" / output_name
            output_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_file, output_image)

    write_dataset_yaml(save_dir / "data.yaml", names)
    print(f"Done. Converted {converted:g}/{len(json_files):g} LabelMe JSON files to {save_dir}")
    return save_dir


def safe_relative_path(path, fallback):
    """Returns a safe relative path for writing inside an output directory."""
    raw = str(path or fallback).replace("\\", "/")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        return Path(path.name or fallback)
    parts = [part for part in path.parts if part not in {"", "."}]
    return Path(*parts) if parts else Path(fallback)


def labelme_image_file(json_file, image_name):
    """Returns the local image path referenced by a LabelMe JSON file."""
    path = Path(str(image_name).replace("\\", "/"))
    return path if path.is_absolute() else json_file.parent / path


def labelme_points(shape, width, height):
    """Returns LabelMe shape points, including decoded mask contours when available."""
    points = shape.get("points", [])
    if shape.get("shape_type") == "mask" and shape.get("mask"):
        mask_points = mask2points(shape["mask"], points, width, height)
        if mask_points:
            return mask_points
    if shape.get("shape_type") == "circle" and len(points) >= 2:
        center, edge = np.array(points[0], dtype=np.float64), np.array(points[1], dtype=np.float64)
        r = np.linalg.norm(edge - center)
        return [[center[0] - r, center[1] - r], [center[0] + r, center[1] + r]]
    return points


def mask2points(mask_data, points, width, height):
    """Converts a LabelMe base64 mask to polygon points."""
    mask = np.array(Image.open(io.BytesIO(base64.b64decode(mask_data))).convert("L"))
    if mask.shape[:2] != (height, width) and len(points):
        full_mask = np.zeros((height, width), dtype=np.uint8)
        x, y = np.array(points, dtype=np.float64).min(axis=0).astype(int)
        h, w = mask.shape[:2]
        full_mask[y : y + h, x : x + w] = mask
        mask = full_mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    return cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True).reshape(-1, 2).tolist()


def yolo_bbox(points, width, height):
    """Converts shape points to normalized YOLO xywh."""
    points = np.array(points, dtype=np.float64).reshape(-1, 2)
    x, y = points[:, 0], points[:, 1]
    box = [
        (x.min() + x.max()) / 2 / width,
        (y.min() + y.max()) / 2 / height,
        (x.max() - x.min()) / width,
        (y.max() - y.min()) / height,
    ]
    return box if box[2] > 0 and box[3] > 0 else []


def yolo_segment(points, width, height):
    """Converts polygon points to normalized YOLO segmentation coordinates."""
    points = np.array(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return []
    return (points / np.array([width, height])).reshape(-1).tolist()


def write_dataset_yaml(file, names):
    """Writes a YOLO dataset YAML with class names."""
    names = names if isinstance(names, dict) else dict(enumerate(names))
    with open(file, "w") as f:
        yaml.safe_dump(
            {"path": str(file.parent), "train": "images", "val": "images", "names": names},
            f,
            sort_keys=False,
        )


def bbox_from_keypoints(ann):
    """Creates a COCO xywh box from visible keypoints when bbox is missing."""
    keypoints = np.array(ann.get("keypoints", []), dtype=np.float64).reshape(-1, 3)
    visible = keypoints[keypoints[:, 2] > 0]
    if not len(visible):
        return [0, 0, 0, 0]
    x, y = visible[:, 0], visible[:, 1]
    return [x.min(), y.min(), x.max() - x.min(), y.max() - y.min()]


def coco_keypoints(ann, width, height):
    """Normalizes COCO keypoints to YOLO pose format."""
    keypoints = ann.get("keypoints", [])
    return (np.array(keypoints, dtype=np.float64).reshape(-1, 3) / np.array([width, height, 1])).reshape(-1).tolist()


def rle2polygon(segmentation):
    """Converts COCO RLE segmentation to polygon segments."""
    from pycocotools import mask

    if isinstance(segmentation["counts"], list):
        segmentation = mask.frPyObjects(segmentation, *segmentation["size"])
    m = mask.decode(segmentation)
    m[m > 0] = 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    return [cv2.approxPolyDP(c, 0.001 * cv2.arcLength(c, True), True).flatten().tolist() for c in contours]


def write_coco_yaml(file, data, coco80, cls91to80):
    """Writes class id-to-name metadata from COCO categories."""
    names = coco_names(data, coco80, cls91to80)
    if names:
        with open(file, "w") as f:
            yaml.safe_dump({"names": names}, f, sort_keys=False)


def coco_names(data, coco80, cls91to80):
    """Returns class id-to-name metadata from COCO categories."""
    names = {}
    for category in data.get("categories", []):
        class_id = coco80[category["id"] - 1] if cls91to80 else category["id"] - 1
        if class_id is not None:
            names[int(class_id)] = category["name"]
    return dict(sorted(names.items()))


def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).

    Returns:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file. like [segmentation1, segmentation2,...], each
            segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def delete_dsstore(path="../datasets"):
    """Deletes Apple .DS_Store files recursively from a specified directory."""
    from pathlib import Path

    files = list(Path(path).rglob(".DS_store"))
    print(files)
    for f in files:
        f.unlink()


def parse_args():
    """Parses command-line arguments for legacy standalone conversion."""
    parser = argparse.ArgumentParser(description="Convert JSON annotations to YOLO labels.")
    parser.add_argument(
        "--source", default="COCO", choices=["COCO", "LabelMe", "infolks", "vott", "ath"], help="Input format."
    )
    parser.add_argument("--json-dir", default="../datasets/coco/annotations", help="Directory containing JSON files.")
    parser.add_argument("--save-dir", default="new_dir", help="Output directory.")
    parser.add_argument("--use-segments", action="store_true", help="Export COCO segmentation labels.")
    parser.add_argument("--use-keypoints", action="store_true", help="Export COCO keypoint labels.")
    parser.add_argument("--cls91to80", action="store_true", help="Map COCO 91-category ids to 80-category ids.")
    parser.add_argument("--name", default="out", help="Output stem for INFOLKS and VoTT text files.")
    parser.add_argument("--files", default="../data/sm4/json/*.json", help="Input JSON glob for INFOLKS and VoTT.")
    parser.add_argument("--img-path", default="../data/sm4/images/", help="Image directory for INFOLKS and VoTT.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.source == "COCO":
        convert_coco_json(args.json_dir, args.use_segments, args.use_keypoints, args.cls91to80, args.save_dir)
    elif args.source == "LabelMe":
        convert_labelme_json(args.json_dir, args.use_segments, args.save_dir)
    elif args.source == "infolks":
        convert_infolks_json(name=args.name, files=args.files, img_path=args.img_path, save_dir=args.save_dir)
    elif args.source == "vott":
        convert_vott_json(name=args.name, files=args.files, img_path=args.img_path, save_dir=args.save_dir)
    elif args.source == "ath":
        convert_ath_json(json_dir=args.json_dir, save_dir=args.save_dir)

    # zip results
    # os.system('zip -r ../coco.zip ../coco')
