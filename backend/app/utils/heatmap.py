from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

def normalize_heatmap(heatmap):
    heatmap = np.nan_to_num(heatmap)
    heatmap = heatmap - heatmap.min()
    max_value = heatmap.max()
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap

def gradient_saliency_heatmap(model, model_input, class_idx=None):
    x = tf.convert_to_tensor(model_input, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]).numpy())
        score = preds[:, class_idx]

    grads = tape.gradient(score, x)
    if grads is None:
        return None

    saliency = tf.reduce_mean(tf.abs(grads), axis=-1)[0].numpy()
    return normalize_heatmap(saliency)

def overlay_heatmap_on_rgb(img_rgb, heatmap, alpha=0.42):
    if heatmap is None:
        return img_rgb

    heatmap = normalize_heatmap(heatmap)
    heatmap_u8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    if heatmap_rgb.shape[:2] != img_rgb.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb, (img_rgb.shape[1], img_rgb.shape[0]))

    overlay = (1 - alpha) * img_rgb.astype("float32") + alpha * heatmap_rgb.astype("float32")
    overlay = np.clip(overlay, 0, 255).astype("uint8")
    return overlay

def save_heatmap_overlay(img_rgb, heatmap, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay = overlay_heatmap_on_rgb(img_rgb, heatmap)
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
