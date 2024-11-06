import json
import os
import re
from argparse import ArgumentParser

import cv2
import numpy as np
from PIL import Image
from loguru import logger
from matplotlib import pyplot as plt
from pycocotools import mask as mask_utils

class AnnotationProcessor:
    def __init__(self, geojson_path, folder, ann_folder, img_folder):
        with open(geojson_path, 'r') as f:
            self.geo = json.load(f)
        self.features = self.geo.get('features', [])
        self.folder = folder
        self.ann_folder = ann_folder
        self.img_folder = img_folder

    def generate_annotation(self, feature, x, y):
        objects = []
        coords = feature.get('geometry', {}).get('coordinates', [])
        mark = coords[0][0]
        if x < mark[0] < x + 4096 and y < mark[1] < y + 4096:
            idx = 1 if feature.get("properties", {}).get("classification") is not None else 0
            color = (0, 255, 0) if feature.get("properties", {}).get("classification") is not None else (0, 0, 255)

            for coord in coords:
                coord_points = [[(a - x) / 4, (b - y) / 4] for a, b in coord]
                coord_points_img = [[(a - x), (b - y)] for a, b in coord]
                cnt = np.array(coord_points, dtype=np.int32).reshape(-1, 1, 2)
                image = np.zeros((1024, 1024), dtype=np.uint8)

                img_contours = [np.array(coord_points_img, dtype=np.int32).reshape(-1, 1, 2)]
                cv2.drawContours(img, img_contours, -1, color, 2)
                cv2.drawContours(image, [cnt], -1, 255, 2)
                rle = mask_utils.encode(np.array(image[:, :, None], order='F', dtype="uint8"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")

                rect = cv2.minAreaRect(cnt)
                (center_x, center_y), (width, height) = rect[0], rect[1]

                obj = {
                    "id": idx,
                    "category_id": idx,
                    "segmentation": rle,
                    "bbox": [cv2.boundingRect(cnt)],
                    "area": int(cv2.contourArea(cnt)),
                    "predicted_iou": 1,
                    "stability_score": 1,
                    "crop_box": [int(center_x - width / 2), int(center_y - height / 2), width, height],
                    "point_coords": coord_points,
                }
                objects.append(obj)
        return objects

    def show_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()

    def process_image(self, file_name):
        if 'mx1631040.' not in file_name or 'CKpan' in file_name:
            return
        objects = []
        match = re.search(r'_(\d+)_(\d+)\.jpg', file_name)
        x_str, y_str = match.groups()
        x, y = int(x_str), int(y_str)
        img = cv2.imread(os.path.join(self.folder, file_name))

        for feature in self.features:
            if feature.get('geometry', {}).get('type') == "Polygon":
                objects.extend(self.generate_annotation(feature, x, y))

        self.show_image(img)
        if len(objects) == 0:
            return

        annotation = {
            "image": {
                "image_id": "",
                "folder": self.folder,
                "file_name": file_name,
                "width": 4096,
                "height": 4096,
            },
            "annotations": objects
        }
        os.makedirs(self.ann_folder, exist_ok=True)
        os.makedirs(self.img_folder, exist_ok=True)
        bn = os.path.splitext(file_name)[0]
        with open(os.path.join(self.ann_folder, f'mx1631040_{x}_{y}.json'), 'w') as f:
            json.dump(annotation, f, indent=2)
        logger.info(f'process {file_name}')
        img = Image.open(os.path.join(self.folder, file_name))
        resized_img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
        resized_img.save(os.path.join(self.img_folder, f'mx1631040_{x}_{y}.jpg'))

    def run(self):
        file_names = os.listdir(self.folder)
        for fn in file_names:
            self.process_image(fn)

parser = ArgumentParser()
parser.add_argument('--geojson_path', type=int, default=-1, help='')
parser.add_argument('--image_dir', type=bool, default=True)
parser.add_argument('--annotation_dir', type=str, default='IHC2HE')
parser.add_argument('--', type=str, default='CKpan')

if __name__ == "__main__":
    geojson_path = r'/data2/yhhu/LLB/Data/前列腺癌数据/DAB染色/qpdata/mx1631040.2-HE.geojson'
    folder = r'/data2/yhhu/LLB/Data/前列腺癌数据/DAB染色/patch/4096/image/'
    ann_folder = r'/data2/yhhu/LLB/Data/前列腺癌数据/DAB染色/patch/4096/seg_train/annotation/'
    img_folder = r'/data2/yhhu/LLB/Data/前列腺癌数据/DAB染色/patch/4096/seg_train/image/'
    processor = AnnotationProcessor(geojson_path, folder, ann_folder, img_folder)
    processor.run()