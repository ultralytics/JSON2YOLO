**Below we will list each of the formats and their corresponding JSON formats**

## INFOLKS Annotation Format

Source: Email reply from folks at INFOLKS
```
{
    "path": "Sample_1.png",
    "output": {
        "objects": [
            {
                "name": "Car_16801237862",
                "description": {
                    "type": "types.default.bndbox"
                },
                "classTitle": "Car",
                "attributes": {},
                "points": {
                    "exterior": [
                        [
                            133.23622545915137,
                            59.66022799240026
                        ],
                        [
                            285.3394553514883,
                            183.11621279290694
                        ]
                    ],
                    "interior": []
                }
            },
            {
                "name": "Car_16801237863",
                "description": {
                    "type": "types.default.contour"
                },
                "classTitle": "Car",
                "attributes": {},
                "points": {
                    "exterior": [
                        [
                            496.7834072197594,
                            206.30683977200763
                        ],
                        [
                            582.7251424952502,
                            151.05858138062064
                        ],
                        [
                            756.6548448385055,
                            150.37650411652945
                        ],
                        [
                            825.5446485117163,
                            180.38790373654214
                        ],
                        [
                            884.8853704876506,
                            217.90215326155797
                        ],
                        [
                            906.7118429385688,
                            285.42780240658647
                        ],
                        [
                            909.4401519949336,
                            387.739392020266
                        ],
                        [
                            724.5972134262191,
                            449.1263457884737
                        ],
                        [
                            627.7422419252692,
                            451.1725775807473
                        ],
                        [
                            534.2976567447753,
                            439.577264091197
                        ],
                        [
                            455.8587713742876,
                            400.0167827739075
                        ],
                        [
                            440.17099430019005,
                            352.95345155161493
                        ],
                        [
                            427.8936035465485,
                            285.42780240658647
                        ],
                        [
                            438.12476250791644,
                            234.2720075997467
                        ]
                    ],
                    "interior": []
                }
            }
        ],
        "image": {
            "name": "Sample_1",
            "width": 1057,
            "height": 592,
            "attributes": {}
        }
    },
    "time_labeled": 1644059585978,
    "labeled": true
}
```

## VOTT Annotation FORMAT
Source:  https://roboflow.com/formats/vott-json

```
{
    "asset": {
        "format": "jpg",
        "id": "0a2ac9053d4d842653d3ff9f988421a6",
        "name": "img0001.jpg",
        "path": "file:D:/HardHats/img0001.jpg",
        "size": {
            "width": 612,
            "height": 408
        },
        "state": 2,
        "type": 1
    },
    "regions": [
        {
            "id": "XEhNEKjZT",
            "type": "RECTANGLE",
            "tags": [
                "helmet"
            ],
            "boundingBox": {
                "height": 204,
                "width": 505.5652173913043,
                "left": 32.06688963210702,
                "top": 143.9598662207358
            },
            "points": [
                {
                    "x": 32.06688963210702,
                    "y": 143.9598662207358
                },
                {
                    "x": 537.6321070234113,
                    "y": 143.9598662207358
                },
                {
                    "x": 537.6321070234113,
                    "y": 347.95986622073576
                },
                {
                    "x": 32.06688963210702,
                    "y": 347.95986622073576
                }
            ]
        }
    ],
    "version": "2.1.0"
}
```

## COCO Annotation FORMAT
Source: https://roboflow.com/formats/coco-json

```
{
    "info": {
        "year": "2020",
        "version": "1",
        "description": "Exported from roboflow.ai",
        "contributor": "Roboflow",
        "url": "https://app.roboflow.ai/datasets/hard-hat-sample/1",
        "date_created": "2000-01-01T00:00:00+00:00"
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "name": "Public Domain"
        }
    ],s
    "categories": [
        {
            "id": 0,
            "name": "Workers",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "head",
            "supercategory": "Workers"
        },
        {
            "id": 2,
            "name": "helmet",
            "supercategory": "Workers"
        },
        {
            "id": 3,
            "name": "person",
            "supercategory": "Workers"
        }
    ],
    "images": [
        {
            "id": 0,
            "license": 1,
            "file_name": "0001.jpg",
            "height": 275,
            "width": 490,
            "date_captured": "2020-07-20T19:39:26+00:00"
        }
    ],
    "annotations": [
        {
            "id": 0,
            "image_id": 0,
            "category_id": 2,
            "bbox": [
                45,
                2,
                85,
                85
            ],
            "area": 7225,
            "segmentation": [],
            "iscrowd": 0
        },
        {
            "id": 1,
            "image_id": 0,
            "category_id": 2,
            "bbox": [
                324,
                29,
                72,
                81
            ],
            "area": 5832,
            "segmentation": [],
            "iscrowd": 0
        }
    ]
}
```

## YOLO FORMAT
- YOLO uses .txt file, one for each image. One row for one label

Source: https://roboflow.com/formats/yolo-darknet-txt

Below, we can see the structure of YOLO Darknet TXT.

Each image has one txt file with a single line for each bounding box. The format of each row is:
```
class_id center_x center_y width height
```
FileName: ```img0001.txt```
File Contents:
```
1 0.408 0.30266666666666664 0.104 0.15733333333333333
1 0.245 0.424 0.046 0.08
```
FileName: ```darknet.labels```
File Contents:
```
head
helmet
person
```




