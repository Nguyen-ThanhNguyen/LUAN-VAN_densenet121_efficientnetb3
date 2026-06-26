from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input


def read_image_rgb(image_path):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_black_border_rgb(img_rgb, tol=7):
    if img_rgb is None or img_rgb.size == 0:
        return img_rgb

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > tol

    if mask.sum() < 100:
        return img_rgb

    ys, xs = np.where(mask)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    cropped = img_rgb[y1:y2 + 1, x1:x2 + 1]

    if cropped.size == 0:
        return img_rgb
    return cropped


def apply_clahe_lab_rgb(img_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def preprocess_fundus_image(image_path, img_size=320, use_clahe=True):
    img_rgb = read_image_rgb(image_path)
    img_rgb = crop_black_border_rgb(img_rgb, tol=7)

    if use_clahe:
        img_rgb = apply_clahe_lab_rgb(img_rgb, clip_limit=2.0, tile_grid_size=(8, 8))

    img_rgb = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img_rgb


def save_rgb_image(img_rgb, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def prepare_model_input(img_rgb, model_type):
    x = img_rgb.astype("float32")
    x = np.expand_dims(x, axis=0)

    model_type = model_type.lower()
    if model_type == "densenet121":
        return densenet_preprocess_input(x)
    if model_type == "efficientnetb3":
        return efficientnet_preprocess_input(x)

    raise ValueError(f"Không hỗ trợ model_type: {model_type}")
