import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from loguru import logger
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

device = torch.device(f"cuda:{4}")

logger.info(f"using device: {device}")

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

np.random.seed(3)


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


sam2_checkpoint = "../sam2_logs/small/checkpoints/checkpoint.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
# sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.5,
    stability_score_offset=0.5,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    use_m2m=True,
)
image = Image.open('/data2/yhhu/LLB/Data/前列腺癌数据/DAB染色/patch/4096/sam2/image/mx1631040.2-HE-CKpan_36864_28672.jpg')
image = np.array(image.convert("RGB"))
plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.axis('off')
plt.show()
# target_image = Image.open(
#     '/data2/yhhu/LLB/Data/前列腺癌数据/DAB染色/patch/4096/seg_train/sam2/mx1631040_73728_32768.jpg')
# target_image = np.array(target_image.convert("RGB"))
# plt.figure(figsize=(20, 20))
# plt.imshow(target_image)
# plt.axis('off')
# plt.show()
masks = mask_generator.generate(image)
plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
