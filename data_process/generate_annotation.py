import json
import os
import re
import traceback
import uuid
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from pycocotools import mask as mask_utils


def affine_transform(points, a, b, c, d, e, f):
    """
    Apply an affine transformation to a set of 2D points.

    Parameters:
    - points: The input points (numpy array of shape [n, 2]).
    - a, b, c, d, e, f: The affine transformation parameters.

    Returns:
    - transformed_points: The transformed points (numpy array of shape [n, 2]).
    """
    x_new = a * points[:, 0] + b * points[:, 1] + c
    y_new = d * points[:, 0] + e * points[:, 1] + f

    return np.column_stack((x_new, y_new)).flatten()


class AnnotationProcessor:
    def __init__(self, opt):
        self.geojson_path = opt.geojson_path
        self.image_dir = opt.image_dir
        self.patch_size = opt.patch_size
        self.mask_size = opt.mask_size
        self.transfer_path = opt.transfer_path
        self.transfer_key = opt.transfer_key

        self.output_dir = opt.output_dir

    def get_contours(self, cancer=False):
        with open(self.geojson_path, 'r') as f:
            geo_data = json.load(f)
        features = geo_data.get('features', [])
        contours = []

        for feature in features:
            classification = feature.get("properties", {}).get("classification")
            if (cancer and classification is None) or (not cancer and classification is not None):
                coordinates = feature.get('geometry', {}).get('coordinates', [])
                coords = [np.array(coord, dtype=np.int32).reshape(-1, 2) for coord in coordinates]
                contours.extend(coords)

        return contours

    def transfer(self, contours):
        # 将轮廓通过仿射变换映射到另一个染色上
        with open(self.transfer_path, 'r') as f:
            reg_params = json.load(f)

        a, b, c, d, e, f = reg_params[self.transfer_key]
        affine_contours = []
        for cnt in contours:
            affine_cnt = affine_transform(cnt, a, b, c, d, e, f)
            affine_cnt = np.reshape(affine_cnt, (len(affine_cnt) // 2, 2))
            affine_cnt = np.round(affine_cnt * 100) / 100
            affine_contours.append(affine_cnt)

        affine_contours = [cnt.reshape(-1, 1, 2) for cnt in affine_contours]
        return affine_contours

    @staticmethod
    def match(image_name, contours):
        # 将轮廓与图片匹配
        match = re.search(r'_(\d+)_(\d+)\.jpg', image_name)
        x_str, y_str = match.groups()
        x, y = int(x_str), int(y_str)
        matched = []
        for cnt in contours:
            first_coord = cnt[0][0]
            if x < first_coord[0] < x + 4096 and y < first_coord[1] < y + 4096:
                cnt[:, :, 0] -= x
                cnt[:, :, 1] -= y
                matched.append(cnt)
        return matched

    def resize(self, image, contour: list[np.ndarray]):
        times = self.patch_size / self.mask_size
        image = cv2.resize(image, (self.mask_size, self.mask_size), interpolation=cv2.INTER_AREA)
        contours = [(cnt / times).astype(np.int32) for cnt in contour]
        return image, contours

    def generate(self, image_name: str, contours: list[np.ndarray]):
        # save mask annotation
        try:
            contours = self.match(image_name, contours)
            if not len(contours):
                logger.info(f'{image_name} has no contours !!!')
                return

            img = cv2.imread(os.path.join(self.image_dir, image_name))
            image, contours = self.resize(img, contours)

            anns = []
            overall_mask = np.zeros((self.mask_size, self.mask_size), dtype=np.uint8)
            cv2.drawContours(overall_mask, contours, -1, 255, cv2.FILLED)
            for cnt in contours:
                mask_canvas = np.zeros((self.mask_size, self.mask_size), dtype=np.uint8)
                cv2.drawContours(mask_canvas, [cnt], -1, 255, 2)
                rle = mask_utils.encode(np.asfortranarray(mask_canvas))
                rle["counts"] = rle["counts"].decode("utf-8")

                x, y, w, h = cv2.boundingRect(np.array(cnt))
                bbox = [x + w // 2, y + h // 2, w, h]

                area = int(cv2.contourArea(cnt))

                anns.append({
                    "id": uuid.uuid1().hex,  # Annotation id
                    "segmentation": rle,  # Mask saved in COCO RLE format.
                    "bbox": bbox,  # The box around the mask, in XYWH format
                    "area": area,  # The area in pixels of the mask
                    "predicted_iou": 0.8,  # The model's own prediction of the mask's quality
                    "stability_score": 0.8,  # A measure of the mask's quality
                    "crop_box": [self.mask_size // 2, self.mask_size // 2, self.mask_size, self.mask_size],  # The crop of the image used to generate the mask, in XYWH format
                    "point_coords": [cnt.tolist()],  # The point coordinates input to the model to generate the mask
                })
            result = {
                "image": {
                    "image_id": uuid.uuid1().hex,
                    "folder": self.image_dir,
                    "file_name": image_name,
                    "width": self.mask_size,
                    "height": self.mask_size,
                },
                "annotations": anns
            }
            ann_dir, img_dir, mask_dir = os.path.join(self.output_dir, 'annotations'), os.path.join(self.output_dir, 'image'), os.path.join(self.output_dir, 'mask')
            os.makedirs(ann_dir, exist_ok=True)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            with open(os.path.join(ann_dir, image_name.replace('.jpg', '.json')), 'w') as f:
                json.dump(result, f, indent=2)

            cv2.imwrite(os.path.join(img_dir, image_name), image)
            cv2.imwrite(os.path.join(mask_dir, image_name), overall_mask)
            logger.info(f'process {image_name} finished !!!')
        except:
            traceback.print_exc()

    def run(self):
        contours = self.get_contours(True)
        contours = self.transfer(contours)

        images = os.listdir(self.image_dir)
        images = [img for img in images if 'mx1631040.2-HE-CKpan' in img]
        with ThreadPoolExecutor(max_workers=20) as executor:
            for img in images:
                executor.submit(self.generate, img, contours)


parser = ArgumentParser()
parser.add_argument('--geojson_path', type=str, default=r'/data2/yhhu/LLB/Data/前列腺癌数据/DAB染色/qpdata/mx1631040.2-HE.geojson', help='')
parser.add_argument('--image_dir', type=str, default=r'/data2/yhhu/LLB/Data/前列腺癌数据/DAB染色/patch/4096/image/')
parser.add_argument('--patch_size', type=int, default=4096)
parser.add_argument('--mask_size', type=int, default=1024)
parser.add_argument('--transfer', type=bool, default=True)
parser.add_argument('--transfer_path', type=str, default=r'/data2/yhhu/LLB/Data/前列腺癌数据/DAB染色/transform/mx1631040.2-HE-HE2IHC.json')
parser.add_argument('--transfer_key', type=str, default='mx1631040.2-HE-CKpan.svs')
parser.add_argument('--output_dir', type=str, default=r'/data2/yhhu/LLB/Data/前列腺癌数据/DAB染色/patch/4096/sam2/')

if __name__ == "__main__":
    args = parser.parse_args()
    processor = AnnotationProcessor(args)
    processor.run()
