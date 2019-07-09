import json
import glob
from PIL import Image, ExifTags
from pathlib import Path

from tqdm import tqdm

from utils import *


# Convert Labelbox JSON file into YOLO-format labels ---------------------------
def convert_labelbox_json(name, file):
    # Import json
    with open(file) as f:
        data = json.load(f)

    # Create folders
    path = make_folders(name)

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

        # The Labelbox bounding box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'])
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= width[i]  # normalize x
        box[[1, 3]] /= height[i]  # normalize y

        with open('out/labels/' + label_name, 'a') as file:
            file.write('%g %.6f %.6f %.6f %.6f\n' % (x['category_id'] - 1, *box))

    # Split data into train, test, and validate files
    split_files(name, file_name)
    print('Done. Output saved to %s' % (os.getcwd() + os.sep + path))


# Convert INFOLKS JSON file into YOLO-format labels ----------------------------
def convert_infolks_json(name, files, img_path):
    # Create folders
    path = make_folders(name)

    # Import json
    data = []
    for file in glob.glob(files):
        with open(file) as f:
            jdata = json.load(f)
            jdata['json_file'] = file
            data.append(jdata)

    # Write images and shapes
    name = path + os.sep + name
    file_id, file_name, width, height, cat = [], [], [], [], []
    for i, x in enumerate(tqdm(data, desc='Files and Shapes')):
        file_id.append(i)
        f = glob.glob(img_path + Path(x['json_file']).stem + '.*')[0]
        file_name.append(f)
        img = Image.open(f)

        s = exif_size(img)  # (width, height)
        width.append(s[0])
        height.append(s[1])

        # filename
        with open(name + '.txt', 'a') as file:
            file.write('%s\n' % f)

        # categories
        cat.extend(a['classTitle'] for a in x['output']['objects'])

    # Write *.names file
    names = sorted(np.unique(cat))
    with open(name + '.names', 'a') as file:
        [file.write('%s\n' % a) for a in names]

    # Write labels file
    for i, x in enumerate(tqdm(data, desc='Annotations')):
        label_name = Path(file_name[i]).stem + '.txt'

        with open(path + '/labels/' + label_name, 'a') as file:
            for a in x['output']['objects']:
                category_id = names.index(a['classTitle'])

                # The INFOLKS bounding box format is [x-min, y-min, x-max, y-max]
                box = np.array(a['points']['exterior']).ravel()
                box[[0, 2]] /= width[i]  # normalize x
                box[[1, 3]] /= height[i]  # normalize y
                box = [box[[0, 2]].mean(), box[[1, 3]].mean(), box[2] - box[0], box[3] - box[1]]  # xywh

                file.write('%g %.6f %.6f %.6f %.6f\n' % (category_id, *box))

    # Split data into train, test, and validate files
    split_files(name, file_name)
    print('Done. Output saved to %s' % (os.getcwd() + os.sep + path))


# Convert vott JSON file into YOLO-format labels ----------------------------------
def convert_vott_json(name, files, img_path):
    # Create folders
    path = make_folders(name)

    # Import json
    data = []
    for file in glob.glob(files):
        with open(file) as f:
            jdata = json.load(f)
            jdata['json_file'] = file
            data.append(jdata)

    # Write images and shapes
    name = path + os.sep + name
    file_id, file_name, width, height, cat = [], [], [], [], []
    for i, x in enumerate(tqdm(data, desc='Files and Shapes')):
        file_id.append(i)
        file_name.append(Path(x['json_file']).name)
        f = glob.glob(img_path + Path(file_name[i]).stem + '.*')[0]
        img = Image.open(f)

        s = img.size  # width, height
        try:
            exif = dict(img._getexif().items())
            if exif[orientation] == 6:  # rotation 270
                s = (s[1], s[0])
            elif exif[orientation] == 8:  # rotation 90
                s = (s[1], s[0])
        except:
            None

        width.append(s[0])
        height.append(s[1])

        # filename
        with open(name + '.txt', 'a') as file:
            file.write('%s\n' % f)

        # categories
        cat.extend(a['classTitle'] for a in x['output']['objects'])

    # Write *.names file
    names = sorted(np.unique(cat))
    with open(name + '.names', 'a') as file:
        [file.write('%s\n' % a) for a in names]

    # Write labels file
    for i, x in enumerate(tqdm(data, desc='Annotations')):
        image_name = file_name[i]

        extension = image_name.split('.')[-1]
        label_name = image_name.replace(extension, 'txt')

        with open(path + '/labels/' + label_name, 'a') as file:
            for a in x['output']['objects']:
                category_id = names.index(a['classTitle'])

                # The INFOLKS bounding box format is [x-min, y-min, x-max, y-max]
                box = np.array(a['points']['exterior']).ravel()
                box[[0, 2]] /= width[i]  # normalize x
                box[[1, 3]] /= height[i]  # normalize y
                box = [box[[0, 2]].mean(), box[[1, 3]].mean(), box[2] - box[0], box[3] - box[1]]  # xywh

                file.write('%g %.6f %.6f %.6f %.6f\n' % (category_id, *box))

    # Split data into train, test, and validate files
    file_name = [x.replace(x.split('.')[-1], 'png') for x in file_name]  # use .png for all
    split_files(name, file_name, prefix_path='../' + Path(name).stem + '/images/')
    print('Done. Output saved to %s' % (os.getcwd() + os.sep + path))


if __name__ == '__main__':
    source = 'infolks'

    if source is 'labelbox':  # Labelbox https://labelbox.com/
        convert_labelbox_json(name='supermarket2',
                              file='../supermarket2/export-coco.json')

    elif source is 'infolks':  # Infolks https://infolks.info/
        convert_infolks_json(name='supermarket3',
                             files='../supermarket3/jsons/*.json',
                             img_path='../supermarket3/images/')

    elif source is 'vott':  # VoTT https://github.com/microsoft/VoTT
        convert_vott_json(name='a1',
                          files='../../Downloads/a1/json/*.json',
                          img_path='../../Downloads/a1/images/')
