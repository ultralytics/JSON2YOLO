# Ultralytics YOLO 🚀, AGPL-3.0 license
# Label studio video-frame annotation JSON to YOLO format conversion

from pathlib import Path

import numpy as np
import yaml

# TODO add steps to fetch media (image or video frame) height and width to convert to YOLO
# TODO argparse pass "path"


def main(path):
    # Get all files
    file_path = Path(path)  # path to JSON files
    files = file_path.glob("*.json")

    labels = set()  # track all labels

    for f in files:
        annos = dict()
        j = yaml.safe_load(f.read_text())
        labels = set(labels)  # ensure unique labels
        # Fetch media to add height/width (per file)
        # TODO get single video frame
        iHeight: int = 0  # image or frame height
        iWidth: int = 0  # image or frame width

        for e in j:
            file = e.get("file_upload")
            data = e.get("data")
            annotations = e.get("annotations", [])

            for ann in annotations:
                result = ann.get("result", {})

                # Iterate results
                for res in result:
                    res_type = res.get("type")  # could use this to process different types of annotations
                    v = res.get("value", {})
                    label = v.get("labels", [])
                    label = label[0] if label else None  # assumed single label per annotation
                    _ = labels.add(label) if label else None
                    anno = v.get("sequence")

                    for a in anno:
                        # Get annotation data
                        frame = a.get("frame")
                        x = a.get("x")  # assuming x-center point of bbox
                        y = a.get("y")  # assuming y-center point of bbox
                        width = a.get("width")
                        height = a.get("height")
                        # TODO ensure x and y are bounding box center points
                        xywh = np.array([x, y, width, height])
                        xywhn = (
                            xywh / np.array([iWidth, iHeight, iWidth, iHeight]).round(5) if iWidth and iHeight else None
                        )

                        # Add frame annotations
                        if frame not in annos:
                            annos[frame] = [{"label": label, "xywh": xywh, "xywhn": xywhn}]
                        else:
                            annos[frame].append({"label": label, "xywh": xywh, "xywhn": xywhn})

            for fn, det in annos.items():
                labels = sorted(labels)  # sort labels for indices
                # Write to YOLO format for each frame-number appended to source filename, (-fn)
                with open(f"{Path(file).stem}-{fn}.txt", "w") as f:
                    for d in det:
                        l = d.get("label")
                        b = d.get("xywhn")
                        f.write(
                            f"{labels.index(l)} {b[0]} {b[1]} {b[2]} {b[3]}\n"
                        )  # class x_center y_center width height

                # NOTE file saved when existing (with)


if __name__ == "__main__":
    # TODO argparse
    main(args)
