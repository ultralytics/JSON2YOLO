import contextlib
import json

import cv2
import pandas as pd
from PIL import Image
from collections import defaultdict

from utils import *


# Convert INFOLKS JSON file into YOLO-format labels ----------------------------
def convert_infolks_json(name, files, img_path):
    # Create folders
    path = make_dirs()

    # Import json
    data = []
    for file in glob.glob(files):
        with open(file) as f:
            jdata = json.load(f)
            jdata['json_file'] = file
            data.append(jdata)

    # Write images and shapes
    name = path + os.sep + name
    file_id, file_name, wh, cat = [], [], [], []
    for x in tqdm(data, desc='Files and Shapes'):
        f = glob.glob(img_path + Path(x['json_file']).stem + '.*')[0]
        file_name.append(f)
        wh.append(exif_size(Image.open(f)))  # (width, height)
        cat.extend(a['classTitle'].lower() for a in x['output']['objects'])  # categories

        # filename
        with open(name + '.txt', 'a') as file:
            file.write('%s\n' % f)

    # Write *.names file
    names = sorted(np.unique(cat))
    # names.pop(names.index('Missing product'))  # remove
    with open(name + '.names', 'a') as file:
        [file.write('%s\n' % a) for a in names]

    # Write labels file
    for i, x in enumerate(tqdm(data, desc='Annotations')):
        label_name = Path(file_name[i]).stem + '.txt'

        with open(path + '/labels/' + label_name, 'a') as file:
            for a in x['output']['objects']:
                # if a['classTitle'] == 'Missing product':
                #    continue  # skip

                category_id = names.index(a['classTitle'].lower())

                # The INFOLKS bounding box format is [x-min, y-min, x-max, y-max]
                box = np.array(a['points']['exterior'], dtype=np.float32).ravel()
                box[[0, 2]] /= wh[i][0]  # normalize x by width
                box[[1, 3]] /= wh[i][1]  # normalize y by height
                box = [box[[0, 2]].mean(), box[[1, 3]].mean(), box[2] - box[0], box[3] - box[1]]  # xywh
                if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                    file.write('%g %.6f %.6f %.6f %.6f\n' % (category_id, *box))

    # Split data into train, test, and validate files
    split_files(name, file_name)
    write_data_data(name + '.data', nc=len(names))
    print(f'Done. Output saved to {os.getcwd() + os.sep + path}')


# Convert vott JSON file into YOLO-format labels -------------------------------
def convert_vott_json(name, files, img_path):
    # Create folders
    path = make_dirs()
    name = path + os.sep + name

    # Import json
    data = []
    for file in glob.glob(files):
        with open(file) as f:
            jdata = json.load(f)
            jdata['json_file'] = file
            data.append(jdata)

    # Get all categories
    file_name, wh, cat = [], [], []
    for i, x in enumerate(tqdm(data, desc='Files and Shapes')):
        with contextlib.suppress(Exception):
            cat.extend(a['tags'][0] for a in x['regions'])  # categories

    # Write *.names file
    names = sorted(pd.unique(cat))
    with open(name + '.names', 'a') as file:
        [file.write('%s\n' % a) for a in names]

    # Write labels file
    n1, n2 = 0, 0
    missing_images = []
    for i, x in enumerate(tqdm(data, desc='Annotations')):

        f = glob.glob(img_path + x['asset']['name'] + '.jpg')
        if len(f):
            f = f[0]
            file_name.append(f)
            wh = exif_size(Image.open(f))  # (width, height)

            n1 += 1
            if (len(f) > 0) and (wh[0] > 0) and (wh[1] > 0):
                n2 += 1

                # append filename to list
                with open(name + '.txt', 'a') as file:
                    file.write('%s\n' % f)

                # write labelsfile
                label_name = Path(f).stem + '.txt'
                with open(path + '/labels/' + label_name, 'a') as file:
                    for a in x['regions']:
                        category_id = names.index(a['tags'][0])

                        # The INFOLKS bounding box format is [x-min, y-min, x-max, y-max]
                        box = a['boundingBox']
                        box = np.array([box['left'], box['top'], box['width'], box['height']]).ravel()
                        box[[0, 2]] /= wh[0]  # normalize x by width
                        box[[1, 3]] /= wh[1]  # normalize y by height
                        box = [box[0] + box[2] / 2, box[1] + box[3] / 2, box[2], box[3]]  # xywh

                        if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                            file.write('%g %.6f %.6f %.6f %.6f\n' % (category_id, *box))
        else:
            missing_images.append(x['asset']['name'])

    print('Attempted %g json imports, found %g images, imported %g annotations successfully' % (i, n1, n2))
    if len(missing_images):
        print('WARNING, missing images:', missing_images)

    # Split data into train, test, and validate files
    split_files(name, file_name)
    print(f'Done. Output saved to {os.getcwd() + os.sep + path}')


# Convert ath JSON file into YOLO-format labels --------------------------------
def convert_ath_json(json_dir):  # dir contains json annotations and images
    # Create folders
    dir = make_dirs()  # output directory

    jsons = []
    for dirpath, dirnames, filenames in os.walk(json_dir):
        jsons.extend(
            os.path.join(dirpath, filename)
            for filename in [
                f for f in filenames if f.lower().endswith('.json')
            ]
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
        for x in tqdm(data['_via_img_metadata'].values(), desc=f'Processing {json_file}'):
            image_file = str(Path(json_file).parent / x['filename'])
            f = glob.glob(image_file)  # image file
            if len(f):
                f = f[0]
                file_name.append(f)
                wh = exif_size(Image.open(f))  # (width, height)

                n1 += 1  # all images
                if len(f) > 0 and wh[0] > 0 and wh[1] > 0:
                    label_file = dir + 'labels/' + Path(f).stem + '.txt'

                    nlabels = 0
                    try:
                        with open(label_file, 'a') as file:  # write labelsfile
                            # try:
                            #     category_id = int(a['region_attributes']['class'])
                            # except:
                            #     category_id = int(a['region_attributes']['Class'])
                            category_id = 0  # single-class

                            for a in x['regions']:
                                # bounding box format is [x-min, y-min, x-max, y-max]
                                box = a['shape_attributes']
                                box = np.array([box['x'], box['y'], box['width'], box['height']],
                                               dtype=np.float32).ravel()
                                box[[0, 2]] /= wh[0]  # normalize x by width
                                box[[1, 3]] /= wh[1]  # normalize y by height
                                box = [box[0] + box[2] / 2, box[1] + box[3] / 2, box[2],
                                       box[3]]  # xywh (left-top to center x-y)

                                if box[2] > 0. and box[3] > 0.:  # if w > 0 and h > 0
                                    file.write('%g %.6f %.6f %.6f %.6f\n' % (category_id, *box))
                                    n3 += 1
                                    nlabels += 1

                        if nlabels == 0:  # remove non-labelled images from dataset
                            os.system(f'rm {label_file}')
                            # print('no labels for %s' % f)
                            continue  # next file

                        # write image
                        img_size = 4096  # resize to maximum
                        img = cv2.imread(f)  # BGR
                        assert img is not None, 'Image Not Found ' + f
                        r = img_size / max(img.shape)  # size ratio
                        if r < 1:  # downsize if necessary
                            h, w, _ = img.shape
                            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)

                        ifile = dir + 'images/' + Path(f).name
                        if cv2.imwrite(ifile, img):  # if success append image to list
                            with open(dir + 'data.txt', 'a') as file:
                                file.write('%s\n' % ifile)
                            n2 += 1  # correct images

                    except Exception:
                        os.system(f'rm {label_file}')
                        print(f'problem with {f}')

            else:
                missing_images.append(image_file)

    nm = len(missing_images)  # number missing
    print('\nFound %g JSONs with %g labels over %g images. Found %g images, labelled %g images successfully' %
          (len(jsons), n3, n1, n1 - nm, n2))
    if len(missing_images):
        print('WARNING, missing images:', missing_images)

    # Write *.names file
    names = ['knife']  # preserves sort order
    with open(dir + 'data.names', 'w') as f:
        [f.write('%s\n' % a) for a in names]

    # Split data into train, test, and validate files
    split_rows_simple(dir + 'data.txt')
    write_data_data(dir + 'data.data', nc=1)
    print(f'Done. Output saved to {Path(dir).absolute()}')


def convert_coco_json(json_dir='../coco/annotations/', use_segments=False, cls91to80=False):
    save_dir = make_dirs()  # output directory
    coco80 = coco91_to_coco80_class()

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob('*.json')):
        fn = Path(save_dir) / 'labels' / json_file.stem.replace('instances_', '')  # folder name
        fn.mkdir()
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data['annotations']:
            imgToAnns[ann['image_id']].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {json_file}'):
            img = images['%g' % img_id]
            h, w, f = img['height'], img['width'], img['file_name']

            bboxes = []
            segments = []
            for ann in anns:
                if ann['iscrowd']:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = coco80[ann['category_id'] - 1] if cls91to80 else ann['category_id'] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                # Segments
                if use_segments:
                    if len(ann['segmentation']) > 1:
                        s = merge_multi_segment(ann['segmentation'])
                        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                    else:
                        s = [j for i in ann['segmentation'] for j in i]  # all segments concatenated
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                    s = [cls] + s
                    if s not in segments:
                        segments.append(s)

            # Write
            with open((fn / f).with_suffix('.txt'), 'a') as file:
                for i in range(len(bboxes)):
                    line = *(segments[i] if use_segments else bboxes[i]),  # cls, box or segments
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')


def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance. 
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all 
    segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...], 
            each segmentation is a list of coordinates.
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
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def delete_dsstore(path='../datasets'):
    # Delete apple .DS_store files
    from pathlib import Path
    files = list(Path(path).rglob('.DS_store'))
    print(files)
    for f in files:
        f.unlink()


if __name__ == '__main__':
    source = 'COCO'

    if source == 'COCO':
        convert_coco_json('../datasets/coco/annotations',  # directory with *.json
                          use_segments=True,
                          cls91to80=True)

    elif source == 'infolks':  # Infolks https://infolks.info/
        convert_infolks_json(name='out',
                             files='../data/sm4/json/*.json',
                             img_path='../data/sm4/images/')

    elif source == 'vott':  # VoTT https://github.com/microsoft/VoTT
        convert_vott_json(name='data',
                          files='../../Downloads/athena_day/20190715/*.json',
                          img_path='../../Downloads/athena_day/20190715/')  # images folder

    elif source == 'ath':  # ath format
        convert_ath_json(json_dir='../../Downloads/athena/')  # images folder

    # zip results
    # os.system('zip -r ../coco.zip ../coco')
