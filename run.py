import json
import os
import shutil

import numpy as np
from tqdm import tqdm

name = 'yolo'  # new dataset name
file = '../data/coco.json'  # coco json to convert


# Convert COCO JSON file into YOLO format labels -------------------------------
def main(name, file):
    # Import json
    with open(file) as f:
        data = json.load(f)

    # Create folders
    path = 'out'
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
    os.makedirs(path + os.sep + 'labels')  # make new labels folder

    # Write images and shapes
    name = 'out' + os.sep + name
    file_id, file_name, width, height = [], [], [], []
    for i, x in enumerate(tqdm(data['images'], desc='Files and Shapes')):
        file_id.append(x['id'])
        file_name.append('IMG_' + x['file_name'].split('IMG_')[-1])
        width.append(x['width'])
        height.append(x['height'])

        # filename
        with open(name + '.txt', 'a') as file:
            file.write('%s\n' % file_name[i])

        # shapes
        with open(name + '.shapes', 'a') as file:
            file.write('%g, %g\n' % (x['width'], x['height']))

    # Write *.names file
    for x in tqdm(data['categories'], desc='Names'):
        with open(name + '.names', 'a') as file:
            file.write('%s\n' % x['name'])

    # Write labels file
    for x in tqdm(data['annotations'], desc='Annotations'):
        i = file_id.index(x['image_id'])  # image index
        image_name = file_name[i]
        extension = image_name.split('.')[-1]
        label_name = image_name.replace(extension, 'txt')

        # The COCO bounding box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'],dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= width[i]  # normalize x
        box[[1, 3]] /= height[i]  # normalize y

        with open('out/labels/' + label_name, 'a') as file:
            file.write('%g %.6f %.6f %.6f %.6f\n' % (x['category_id'] - 1, *box))

    # Split data into train, test, and validate files
    file_name = sorted(file_name)
    i, j, k = split(file_name, train=0.9, test=0.1, validate=0.0, shuffle=False)
    datasets = {'train': i, 'test': j, 'val': k}
    for key, item in datasets.items():
        with open(name + '_' + key + '.txt', 'a') as file:
            for i in item:
                file.write('%s\n' % file_name[i])

    print('Done. Output saved to %s' % (os.getcwd() + os.sep + path))


def split(x, train=0.9, test=0.1, validate=0.0, shuffle=False):  # split training data
    n = len(x)
    v = np.arange(n)
    if shuffle:
        np.random.shuffle(v)

    i = round(n * train)  # train
    j = round(n * test) + i  # test
    k = round(n * validate) + j  # validate
    return v[:i], v[i:j], v[j:k]


if __name__ == '__main__':
    main(name, file)
