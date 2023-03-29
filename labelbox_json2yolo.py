import json
import os
from pathlib import Path

import requests
import yaml
from PIL import Image
from tqdm import tqdm

from utils import make_dirs

def getPolygons(url):
    # Download the image from the URL
    with urllib.request.urlopen(url) as url_response:
        img_array = np.asarray(bytearray(url_response.read()), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    h, w = image.shape[:2]
    line_width = int((h + w) * 0.5 * 0.0025)
    #convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    contours_approx = []
    polygons = []
    #Get polygons
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)

        contours_approx.append(contour_approx)
        polygon = (contour_approx.flatten().reshape(-1, 2) / np.array([w, h])).tolist()
        polygons.append(polygon)
    cv2.drawContours(image, contours_approx, -1, 128, line_width)
    #format the data correctly
    result = [[f"{x},{y}" for x, y in sublist] for sublist in polygons]
    return [
        [float(x) for pair in sublist for x in pair.split(',')]
        for sublist in result
    ]

def convert(file, zip=True):
    # Convert Labelbox JSON labels to YOLO labels
    names = []  # class names
    file = Path(file)
    save_dir = make_dirs(file.stem)
    with open(file) as f:
        data = json.load(f)  # load JSON

    for img in tqdm(data, desc=f'Converting {file}'):
        im_path = img['Labeled Data']
        im = Image.open(requests.get(im_path, stream=True).raw if im_path.startswith('http') else im_path)  # open
        width, height = im.size  # image size
        label_path = save_dir / 'labels' / Path(img['External ID']).with_suffix('.txt').name
        image_path = save_dir / 'images' / img['External ID']
        im.save(image_path, quality=95, subsampling=0)

     
        ######################################
        #for Image Segmentation 
        ######################################

        # for img in tqdm(data, desc=f'Converting {file}'):
        # im_path = img['Labeled Data']
        # im = Image.open(requests.get(im_path, stream=True).raw if im_path.startswith('http') else im_path)  # open
        # width, height = im.size  # image size
        # label_path =   save_dir / 'labels' / Path(img['External ID']).with_suffix('.txt').name
        # print(label_path)
        # image_path =   save_dir / 'images' / img['External ID']
        # print(image_path)
        # im.save(image_path, quality=95, subsampling=0)

        # for label in img['Label']['objects']:
        #     
        #     polygon = getPolygons(label['instanceURI'])
        #     # class
        #     cls = label['value']  # class name
        #     if cls not in names:
        #         names.append(cls)

        #     line = names.index(cls), *polygon[0]  # YOLO format (class_index, polygon)
        #     with open(label_path, 'a') as f:
        #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

        ######################################
        #for Object detection 
        ######################################
        for label in img['Label']['objects']:
            # box
            top, left, h, w = label['bbox'].values()  # top, left, height, width
            xywh = [(left + w / 2) / width, (top + h / 2) / height, w / width, h / height]  # xywh normalized

            # class
            cls = label['value']  # class name
            if cls not in names:
                names.append(cls)

            line = names.index(cls), *xywh  # YOLO format (class_index, xywh)
            with open(label_path, 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

    # Save dataset.yaml
    d = {'path': f"../datasets/{file.stem}  # dataset root dir",
         'train': "images/train  # train images (relative to path) 128 images",
         'val': "images/val  # val images (relative to path) 128 images",
         'test': " # test images (optional)",
         'nc': len(names),
         'names': names}  # dictionary

    with open(save_dir / file.with_suffix('.yaml').name, 'w') as f:
        yaml.dump(d, f, sort_keys=False)

    # Zip
    if zip:
        print(f'Zipping as {save_dir}.zip...')
        os.system(f'zip -qr {save_dir}.zip {save_dir}')

    print('Conversion completed successfully!')


if __name__ == '__main__':
    convert('export-2021-06-29T15_25_41.934Z.json')
