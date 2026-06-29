
# ==============================================================================
# DR Training Notebook v3 — DenseNet121 + EfficientNetB3 — Imbalance Handling Pipeline
# ==============================================================================

# ==============================================================================
# 1. Kaggle setup và kiểm tra môi trường
# ==============================================================================
from __future__ import annotations

import os
import sys
import json
import math
import random
import shutil
import glob
import time
import subprocess
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("Python:", sys.version)
print("Current working directory:", os.getcwd())
print("/kaggle/input exists:", Path("/kaggle/input").exists())
if Path("/kaggle/input").exists():
    print("/kaggle/input items:", os.listdir("/kaggle/input")[:20])

# ==============================================================================
# 2. Import thư viện
# ==============================================================================
import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from tensorflow.keras.applications import DenseNet121, EfficientNetB3
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
# EfficientNet Keras đã có rescaling/normalization trong model, preprocess_input gần như pass-through.
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))

# ==============================================================================
# 3. Cấu hình toàn cục
# ==============================================================================
NUM_CLASSES = 5
CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

@dataclass
class TrainConfig:
    data_dir: str = "auto"
    output_root: str = "/kaggle/working/training_outputs"
    artifacts_dir: str = "/kaggle/working/backend_artifacts"

    img_size: int = 320
    batch_size: int = 8
    seed: int = 42

    val_split: float = 0.15
    test_split: float = 0.15

    epochs_head: int = 8
    epochs_finetune: int = 30
    head_lr: float = 1e-3
    finetune_lr: float = 2e-5
    min_lr: float = 1e-7

    dropout: float = 0.45
    unfreeze_last_n: int = 180

    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    preprocess_max_side: int = 900
    use_preprocess_cache: bool = True

    label_smoothing: float = 0.03

    # Xử lý mất cân bằng dữ liệu.
    # Khuyến nghị v3: dùng balanced_batch để mỗi epoch model nhìn thấy class 1/3/4 nhiều hơn.
    # Không dùng class_weight đồng thời với balanced_batch mặc định để tránh over-correct.
    imbalance_strategy: str = "balanced_batch"  # options: "none", "class_weight", "balanced_batch"
    use_class_weight: bool = False
    minority_classes: str = "1,3,4"
    minority_aug_boost: bool = True
    balanced_sampling_power: float = 1.0  # 1.0 = lấy lớp gần như đều nhau; 0.5 = nhẹ hơn

    # Loss tùy chọn. Focal loss có thể tăng recall lớp khó nhưng có thể giảm accuracy.
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha_from_class_freq: bool = False

    monitor_metric: str = "val_macro_f1"  # v3 ưu tiên Macro-F1 khi xử lý mất cân bằng
    early_stop_patience: int = 8
    reduce_lr_patience: int = 3

    # Khi True: cố dùng ImageNet pretrained. Nếu Kaggle bị HTTP 403, code tự fallback.
    use_imagenet_weights: bool = True
    # Nếu có dataset weights offline, notebook sẽ tự tìm trong /kaggle/input.
    allow_offline_weights: bool = True
    # Nếu không tải được weights và không có offline weights, vẫn train từ đầu để không dừng notebook.
    allow_random_init_fallback: bool = True

    mixed_precision: bool = True

CFG = TrainConfig()
print(json.dumps(asdict(CFG), indent=2, ensure_ascii=False))

# ==============================================================================
# 4. Hàm tiện ích: seed, thư mục, JSON, TensorFlow
# ==============================================================================
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_tensorflow(cfg: TrainConfig) -> None:
    set_global_seed(cfg.seed)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("Cannot set memory growth:", e)

    if cfg.mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled.")
        except Exception as e:
            print("Mixed precision not enabled:", e)

setup_tensorflow(CFG)
OUTPUT_ROOT = ensure_dir(CFG.output_root)
ARTIFACTS_DIR = ensure_dir(CFG.artifacts_dir)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("ARTIFACTS_DIR:", ARTIFACTS_DIR)

# ==============================================================================
# 5. Tự động tìm dataset APTOS 2019
# ==============================================================================
def find_aptos_data_dir() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/aptos2019-blindness-detection"),
        Path("/kaggle/input/aptos2019-blindness-detection"),
    ]

    for p in candidates:
        if (p / "train.csv").exists() and (p / "train_images").exists():
            return p

    # Tìm rộng hơn trong /kaggle/input
    for train_csv in Path("/kaggle/input").glob("**/train.csv"):
        p = train_csv.parent
        if (p / "train_images").exists():
            return p

    print("Không tìm thấy dataset APTOS. Danh sách /kaggle/input:")
    if Path("/kaggle/input").exists():
        for item in Path("/kaggle/input").iterdir():
            print("-", item)
    raise FileNotFoundError("Không tìm thấy train.csv và train_images trong /kaggle/input.")

DATA_DIR = find_aptos_data_dir() if CFG.data_dir == "auto" else Path(CFG.data_dir)
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMG_DIR = DATA_DIR / "train_images"

print("DATA_DIR =", DATA_DIR)
print("Files:", os.listdir(DATA_DIR))
print("train.csv exists:", TRAIN_CSV.exists())
print("train_images exists:", TRAIN_IMG_DIR.exists())

# ==============================================================================
# 6. Đọc dữ liệu và kiểm tra phân bố nhãn
# ==============================================================================
df = pd.read_csv(TRAIN_CSV)

# Chuẩn hóa tên cột của APTOS: id_code, diagnosis
if "id_code" not in df.columns or "diagnosis" not in df.columns:
    raise ValueError(f"train.csv cần có cột id_code và diagnosis. Columns hiện có: {df.columns.tolist()}")

df["image_path"] = df["id_code"].apply(lambda x: str(TRAIN_IMG_DIR / f"{x}.png"))
df["exists"] = df["image_path"].apply(lambda p: Path(p).exists())
missing = int((~df["exists"]).sum())
print("Số dòng train.csv:", len(df))
print("Số ảnh bị thiếu:", missing)
if missing > 0:
    print(df.loc[~df["exists"], ["id_code", "image_path"]].head())
    df = df[df["exists"]].reset_index(drop=True)

print("Phân bố nhãn:")
print(df["diagnosis"].value_counts().sort_index())

plt.figure(figsize=(7, 4))
df["diagnosis"].value_counts().sort_index().plot(kind="bar")
plt.title("APTOS 2019 - Class distribution")
plt.xlabel("Diagnosis class")
plt.ylabel("Number of images")
plt.show()

# ==============================================================================
# 7. Tiền xử lý ảnh: crop viền đen + CLAHE + resize — Không ESRGAN
# ==============================================================================
def crop_black_borders_rgb(img_rgb: np.ndarray, threshold: int = 7) -> np.ndarray:
    """Crop vùng ảnh có nội dung, loại viền đen quanh ảnh đáy mắt."""
    if img_rgb is None or img_rgb.size == 0:
        return img_rgb
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    if not np.any(mask):
        return img_rgb
    ys, xs = np.where(mask)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    cropped = img_rgb[y1:y2 + 1, x1:x2 + 1]
    if cropped.size == 0:
        return img_rgb
    return cropped


def limit_max_side(img_rgb: np.ndarray, max_side: int = 900) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img_rgb
    scale = max_side / float(longest)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def apply_clahe_lab_rgb(img_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """CLAHE nhẹ trên kênh L của không gian LAB."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def preprocess_image_file(src_path: str | Path, img_size: int, cfg: TrainConfig) -> np.ndarray:
    """Trả ảnh RGB uint8 kích thước img_size x img_size, chưa normalize theo backbone."""
    src_path = str(src_path)
    bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Không đọc được ảnh: {src_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = crop_black_borders_rgb(rgb)
    rgb = limit_max_side(rgb, cfg.preprocess_max_side)
    if cfg.use_clahe:
        rgb = apply_clahe_lab_rgb(rgb, cfg.clahe_clip_limit, cfg.clahe_tile_grid_size)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


def show_preprocess_examples(df_in: pd.DataFrame, n: int = 3) -> None:
    sample = df_in.sample(min(n, len(df_in)), random_state=CFG.seed)
    for _, row in sample.iterrows():
        original_bgr = cv2.imread(row["image_path"])
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        processed = preprocess_image_file(row["image_path"], CFG.img_size, CFG)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(original_rgb)
        ax[0].set_title(f"Original - class {row['diagnosis']}")
        ax[0].axis("off")
        ax[1].imshow(processed)
        ax[1].set_title("Processed")
        ax[1].axis("off")
        plt.show()

show_preprocess_examples(df, n=3)

# ==============================================================================
# 8. Chia train / validation / test theo stratified split
# ==============================================================================
def make_splits(df_all: pd.DataFrame, cfg: TrainConfig) -> Dict[str, pd.DataFrame]:
    df_all = df_all.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    # Tách test trước
    train_val_df, test_df = train_test_split(
        df_all,
        test_size=cfg.test_split,
        random_state=cfg.seed,
        stratify=df_all["diagnosis"],
    )

    # Tách validation từ phần train_val
    val_ratio_in_train_val = cfg.val_split / (1.0 - cfg.test_split)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_in_train_val,
        random_state=cfg.seed,
        stratify=train_val_df["diagnosis"],
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
    return splits

splits = make_splits(df, CFG)
for name, part in splits.items():
    print(f"\n{name}: {len(part)}")
    print(part["diagnosis"].value_counts().sort_index())

SPLIT_DIR = ensure_dir(OUTPUT_ROOT / "splits")
for name, part in splits.items():
    part.to_csv(SPLIT_DIR / f"{name}.csv", index=False)
print("Saved splits to", SPLIT_DIR)

# ==============================================================================
# 9. Chẩn đoán mất cân bằng dữ liệu và cấu hình sampling
# ==============================================================================
def parse_minority_classes(cfg: TrainConfig) -> List[int]:
    vals = []
    for item in str(cfg.minority_classes).split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    return vals


def show_imbalance_report(splits: Dict[str, pd.DataFrame], cfg: TrainConfig) -> None:
    print("Imbalance strategy:", cfg.imbalance_strategy)
    print("use_class_weight:", cfg.use_class_weight)
    print("use_focal_loss:", cfg.use_focal_loss)
    print("monitor_metric:", cfg.monitor_metric)
    print("minority_classes:", parse_minority_classes(cfg))

    rows = []
    for split_name, part in splits.items():
        counts = part["diagnosis"].value_counts().sort_index()
        for c in range(NUM_CLASSES):
            rows.append({"split": split_name, "class": c, "count": int(counts.get(c, 0))})
    rep = pd.DataFrame(rows)
    display(rep.pivot(index="class", columns="split", values="count"))

    train_counts = splits["train"]["diagnosis"].value_counts().sort_index()
    max_count = int(train_counts.max())
    imbalance_ratio = {int(c): float(max_count / max(1, int(train_counts.get(c, 0)))) for c in range(NUM_CLASSES)}
    print("Imbalance ratio trên train, so với lớp nhiều nhất:")
    print(json.dumps(imbalance_ratio, indent=2, ensure_ascii=False))

show_imbalance_report(splits, CFG)

# ==============================================================================
# 10. Tạo cache ảnh đã tiền xử lý
# ==============================================================================
CACHE_DIR = ensure_dir(Path("/kaggle/working") / f"preprocessed_{CFG.img_size}_clahe_{int(CFG.use_clahe)}")
print("CACHE_DIR:", CACHE_DIR)


def cache_one_image(row: pd.Series, cfg: TrainConfig, cache_dir: Path) -> str:
    dst = cache_dir / f"{row['id_code']}.png"
    if dst.exists():
        return str(dst)
    img = preprocess_image_file(row["image_path"], cfg.img_size, cfg)
    # Lưu RGB sang PNG bằng cv2 cần convert sang BGR
    cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return str(dst)


def build_preprocess_cache(df_all: pd.DataFrame, cfg: TrainConfig, cache_dir: Path) -> pd.DataFrame:
    df_out = df_all.copy()
    cache_paths = []
    start = time.time()
    for i, row in df_out.iterrows():
        cache_paths.append(cache_one_image(row, cfg, cache_dir))
        if (i + 1) % 500 == 0:
            print(f"Cached {i + 1}/{len(df_out)} images...")
    df_out["cache_path"] = cache_paths
    print(f"Cache done in {(time.time() - start)/60:.2f} minutes")
    return df_out

if CFG.use_preprocess_cache:
    df_cached = build_preprocess_cache(df, CFG, CACHE_DIR)
else:
    df_cached = df.copy()
    df_cached["cache_path"] = df_cached["image_path"]

# Cập nhật lại split với cache_path
cache_map = dict(zip(df_cached["id_code"], df_cached["cache_path"]))
for name in splits:
    splits[name] = splits[name].copy()
    splits[name]["cache_path"] = splits[name]["id_code"].map(cache_map)
    splits[name].to_csv(SPLIT_DIR / f"{name}.csv", index=False)

# ==============================================================================
# 11. Data Generator chuẩn cho từng backbone
# ==============================================================================
def augment_rgb_uint8(img: np.ndarray, label: Optional[int] = None, cfg: Optional[TrainConfig] = None) -> np.ndarray:
    """Augmentation phù hợp ảnh fundus.

    Với lớp ít dữ liệu 1/3/4, tăng nhẹ xác suất và cường độ augmentation để giảm overfit khi oversampling.
    Không dùng biến đổi quá mạnh vì có thể làm sai đặc trưng bệnh.
    """
    cfg = cfg or CFG
    label = int(label) if label is not None else None
    minority = label in parse_minority_classes(cfg)
    boost = bool(cfg.minority_aug_boost and minority)

    # flip ngang
    if random.random() < (0.55 if boost else 0.5):
        img = np.ascontiguousarray(np.fliplr(img))
    # flip dọc: có thể dùng cho fundus nhưng giữ xác suất vừa phải
    if random.random() < (0.35 if boost else 0.25):
        img = np.ascontiguousarray(np.flipud(img))

    # xoay / zoom nhẹ
    if random.random() < (0.85 if boost else 0.7):
        h, w = img.shape[:2]
        angle = random.uniform(-22, 22) if boost else random.uniform(-15, 15)
        scale = random.uniform(0.90, 1.10) if boost else random.uniform(0.95, 1.05)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # dịch nhẹ
    if boost and random.random() < 0.35:
        h, w = img.shape[:2]
        tx = random.uniform(-0.04, 0.04) * w
        ty = random.uniform(-0.04, 0.04) * h
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # brightness/contrast nhẹ
    if random.random() < (0.65 if boost else 0.5):
        alpha = random.uniform(0.85, 1.15) if boost else random.uniform(0.9, 1.1)
        beta = random.uniform(-12, 12) if boost else random.uniform(-8, 8)
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    return img


def preprocess_for_model(batch_rgb_0_255: np.ndarray, model_type: str) -> np.ndarray:
    x = batch_rgb_0_255.astype(np.float32)
    model_type = model_type.lower()
    if model_type == "densenet121":
        return densenet_preprocess_input(x)
    if model_type == "efficientnetb3":
        # EfficientNet Keras đã có preprocessing trong mô hình; giữ 0-255.
        return efficientnet_preprocess_input(x)
    raise ValueError(f"Unsupported model_type: {model_type}")


def make_class_sampling_probs(train_df: pd.DataFrame, cfg: TrainConfig) -> np.ndarray:
    """Tạo xác suất chọn lớp cho balanced batch.

    power=1.0: gần đều giữa 5 lớp.
    power=0.5: cân bằng nhẹ hơn, giảm nguy cơ over-correct.
    """
    counts = train_df["diagnosis"].value_counts().sort_index()
    inv = []
    for c in range(NUM_CLASSES):
        cnt = max(1, int(counts.get(c, 0)))
        inv.append((1.0 / cnt) ** float(cfg.balanced_sampling_power))
    probs = np.array(inv, dtype=np.float64)
    probs = probs / probs.sum()
    return probs.astype(np.float32)


class DRSequence(Sequence):
    def __init__(self, df_part: pd.DataFrame, cfg: TrainConfig, model_type: str, training: bool):
        self.df = df_part.reset_index(drop=True)
        self.cfg = cfg
        self.model_type = model_type
        self.training = training
        self.indexes = np.arange(len(self.df))
        self.by_class = {
            c: self.df.index[self.df["diagnosis"].astype(int) == c].to_numpy()
            for c in range(NUM_CLASSES)
        }
        self.class_sampling_probs = make_class_sampling_probs(self.df, cfg) if training else None
        self.use_balanced_batch = bool(training and cfg.imbalance_strategy == "balanced_batch")
        self.on_epoch_end()

    def __len__(self):
        # Với balanced batch, giữ số step/epoch xấp xỉ bằng train gốc để không train quá lâu.
        return int(math.ceil(len(self.df) / self.cfg.batch_size))

    def on_epoch_end(self):
        if self.training and not self.use_balanced_batch:
            np.random.shuffle(self.indexes)

    def _sample_balanced_batch(self) -> pd.DataFrame:
        selected_indices = []
        chosen_classes = np.random.choice(
            np.arange(NUM_CLASSES),
            size=self.cfg.batch_size,
            replace=True,
            p=self.class_sampling_probs,
        )
        for c in chosen_classes:
            pool = self.by_class[int(c)]
            if len(pool) == 0:
                selected_indices.append(int(np.random.choice(self.indexes)))
            else:
                selected_indices.append(int(np.random.choice(pool)))
        return self.df.loc[selected_indices]

    def __getitem__(self, idx):
        if self.use_balanced_batch:
            batch = self._sample_balanced_batch()
        else:
            batch_idx = self.indexes[idx * self.cfg.batch_size:(idx + 1) * self.cfg.batch_size]
            batch = self.df.iloc[batch_idx]

        images = []
        labels = []
        for _, row in batch.iterrows():
            label = int(row["diagnosis"])
            path = row.get("cache_path", row["image_path"])
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                img = preprocess_image_file(row["image_path"], self.cfg.img_size, self.cfg)
            else:
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if img.shape[:2] != (self.cfg.img_size, self.cfg.img_size):
                    img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_AREA)
            if self.training:
                img = augment_rgb_uint8(img, label=label, cfg=self.cfg)
            images.append(img)
            labels.append(label)
        x = np.stack(images, axis=0)
        x = preprocess_for_model(x, self.model_type)
        y = to_categorical(np.array(labels), NUM_CLASSES).astype(np.float32)
        return x, y


# Test generator nhanh và kiểm tra class distribution trong vài balanced batches
_gen_test = DRSequence(splits["train"].head(200), CFG, "densenet121", training=True)
x_test, y_test = _gen_test[0]
print("Batch x:", x_test.shape, x_test.dtype, "min/max", float(np.min(x_test)), float(np.max(x_test)))
print("Batch y:", y_test.shape, y_test[:3])

if CFG.imbalance_strategy == "balanced_batch":
    sampled = []
    for i in range(min(20, len(_gen_test))):
        _, yy = _gen_test[i]
        sampled.extend(np.argmax(yy, axis=1).tolist())
    print("Balanced batch sampled class counts trong vài batch đầu:")
    print(pd.Series(sampled).value_counts().sort_index())

# ==============================================================================
# 12. Metrics, callback QWK/Macro-F1 và hàm đánh giá
# ==============================================================================
class MedicalMetricsCallback(Callback):
    """Tính val_qwk và val_macro_f1 cuối mỗi epoch để checkpoint theo metric y tế hơn accuracy."""
    def __init__(self, val_sequence: Sequence, verbose: int = 1):
        super().__init__()
        self.val_sequence = val_sequence
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true, y_pred = predict_sequence_labels(self.model, self.val_sequence)
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        logs["val_qwk"] = qwk
        logs["val_macro_f1"] = macro_f1
        logs["val_macro_recall"] = macro_recall
        if self.verbose:
            print(f" - val_qwk: {qwk:.5f} - val_macro_f1: {macro_f1:.5f} - val_macro_recall: {macro_recall:.5f}")

# Alias để tương thích với tên cũ nếu cần
QWKCallback = MedicalMetricsCallback


def predict_sequence_proba(model: keras.Model, seq: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    probs = model.predict(seq, verbose=0)
    probs = np.asarray(probs, dtype=np.float32)
    y_true = []
    for i in range(len(seq)):
        _, y = seq[i]
        y_true.extend(np.argmax(y, axis=1).tolist())
    y_true = np.array(y_true[:len(probs)], dtype=np.int32)
    return y_true, probs


def predict_sequence_labels(model: keras.Model, seq: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    y_true, probs = predict_sequence_proba(model, seq)
    y_pred = np.argmax(probs, axis=1).astype(np.int32)
    return y_true, y_pred


def compute_auc_ovr(y_true: np.ndarray, probs: np.ndarray) -> Optional[float]:
    try:
        y_oh = to_categorical(y_true, NUM_CLASSES)
        return float(roc_auc_score(y_oh, probs, average="macro", multi_class="ovr"))
    except Exception as e:
        print("AUC skipped:", e)
        return None


def save_confusion_matrix_png(cm: np.ndarray, path: str | Path, title: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, [str(i) for i in range(NUM_CLASSES)])
    plt.yticks(tick_marks, [str(i) for i in range(NUM_CLASSES)])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.show()
    plt.close()


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray, out_dir: Path, prefix: str) -> Dict[str, Any]:
    ensure_dir(out_dir)
    y_pred = np.argmax(probs, axis=1).astype(np.int32)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "auc_macro_ovr": compute_auc_ovr(y_true, probs),
    }

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    save_json(metrics, out_dir / f"{prefix}_metrics.json")
    save_json(report, out_dir / f"{prefix}_classification_report.json")

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })
    for c in range(NUM_CLASSES):
        pred_df[f"prob_{c}"] = probs[:, c]
    pred_df.to_csv(out_dir / f"{prefix}_predictions.csv", index=False)

    pd.DataFrame(cm).to_csv(out_dir / f"{prefix}_confusion_matrix.csv", index=False)
    save_confusion_matrix_png(cm, out_dir / f"{prefix}_confusion_matrix.png", f"{prefix} confusion matrix")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics

# ==============================================================================
# 13. Tìm pretrained weights offline nếu Kaggle chặn tải ImageNet
# ==============================================================================
def find_local_weights_file(model_type: str) -> Optional[str]:
    """Tìm file weights trong /kaggle/input nếu người dùng add dataset chứa weights."""
    if not CFG.allow_offline_weights:
        return None

    model_type = model_type.lower()
    patterns = []
    if model_type == "efficientnetb3":
        patterns = [
            "/kaggle/input/**/*efficientnet*b3*notop*.h5",
            "/kaggle/input/**/*efficientnetb3*.h5",
            "/kaggle/input/**/*efficientnet_b3*.h5",
        ]
    elif model_type == "densenet121":
        patterns = [
            "/kaggle/input/**/*densenet*121*notop*.h5",
            "/kaggle/input/**/*densenet121*.h5",
        ]

    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))
    matches = sorted(set(matches))
    if matches:
        print(f"Found local weights for {model_type}: {matches[0]}")
        return matches[0]
    return None


def create_backbone_with_safe_weights(model_type: str, img_size: int):
    """
    Tạo backbone theo thứ tự:
    1) local offline weights trong /kaggle/input nếu có;
    2) ImageNet online;
    3) weights=None nếu vẫn lỗi.
    """
    model_type = model_type.lower()
    input_shape = (img_size, img_size, 3)

    if model_type == "densenet121":
        factory = DenseNet121
        model_name = "densenet121"
    elif model_type == "efficientnetb3":
        factory = EfficientNetB3
        model_name = "efficientnetb3"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    local_weights = find_local_weights_file(model_type)
    if local_weights:
        try:
            print(f"Building {model_name} with local offline weights: {local_weights}")
            backbone = factory(
                weights=local_weights,
                include_top=False,
                input_shape=input_shape,
                pooling="avg",
            )
            return backbone, "offline_local"
        except Exception as e:
            print(f"Local weights failed for {model_name}: {type(e).__name__}: {e}")

    if CFG.use_imagenet_weights:
        try:
            print(f"Building {model_name} with ImageNet weights...")
            backbone = factory(
                weights="imagenet",
                include_top=False,
                input_shape=input_shape,
                pooling="avg",
            )
            return backbone, "imagenet"
        except Exception as e:
            print(f"Cannot load ImageNet weights for {model_name}: {type(e).__name__}: {e}")

    if CFG.allow_random_init_fallback:
        print(f"Building {model_name} with weights=None. Model will train from scratch.")
        backbone = factory(
            weights=None,
            include_top=False,
            input_shape=input_shape,
            pooling="avg",
        )
        return backbone, "none"

    raise RuntimeError(f"Cannot create {model_name}: no weights available and random fallback disabled.")

# ==============================================================================
# 14. Xây dựng mô hình DenseNet121 / EfficientNetB3
# ==============================================================================
def build_classifier_model(model_type: str, cfg: TrainConfig) -> Tuple[keras.Model, str]:
    backbone, weights_source = create_backbone_with_safe_weights(model_type, cfg.img_size)

    inputs = keras.Input(shape=(cfg.img_size, cfg.img_size, 3), name="input_image")
    x = backbone(inputs, training=False)
    x = layers.Dropout(cfg.dropout, name="dropout")(x)
    # dtype float32 để ổn định khi dùng mixed precision
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{model_type}_dr_classifier")

    return model, weights_source


def set_backbone_trainable(model: keras.Model, trainable: bool, unfreeze_last_n: Optional[int] = None) -> None:
    # backbone là layer thứ 1 sau input trong model
    backbone = None
    for layer in model.layers:
        if "densenet" in layer.name.lower() or "efficientnet" in layer.name.lower():
            backbone = layer
            break
    if backbone is None:
        print("Backbone not found; setting all layers trainable=", trainable)
        for layer in model.layers:
            layer.trainable = trainable
        return

    if not trainable:
        backbone.trainable = False
        return

    backbone.trainable = True
    if unfreeze_last_n is not None and unfreeze_last_n > 0:
        for layer in backbone.layers[:-unfreeze_last_n]:
            layer.trainable = False
        for layer in backbone.layers[-unfreeze_last_n:]:
            # BatchNorm thường nên freeze khi fine-tune để ổn định với batch nhỏ
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True


def categorical_focal_loss(gamma: float = 2.0, alpha: Optional[np.ndarray] = None):
    alpha_tensor = None
    if alpha is not None:
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        focal_factor = tf.pow(1.0 - y_pred, gamma)
        loss = focal_factor * ce
        if alpha_tensor is not None:
            loss = loss * alpha_tensor
        return tf.reduce_sum(loss, axis=-1)

    return loss_fn


def get_focal_alpha_from_train(train_df: pd.DataFrame) -> np.ndarray:
    counts = train_df["diagnosis"].value_counts().sort_index()
    inv = []
    for c in range(NUM_CLASSES):
        inv.append(1.0 / max(1, int(counts.get(c, 0))))
    alpha = np.array(inv, dtype=np.float32)
    alpha = alpha / alpha.sum() * NUM_CLASSES
    return alpha.astype(np.float32)


def compile_model(model: keras.Model, lr: float, cfg: TrainConfig, train_df: Optional[pd.DataFrame] = None) -> None:
    if cfg.use_focal_loss:
        alpha = None
        if cfg.focal_alpha_from_class_freq and train_df is not None:
            alpha = get_focal_alpha_from_train(train_df)
            print("Focal alpha:", alpha)
        loss = categorical_focal_loss(gamma=cfg.focal_gamma, alpha=alpha)
    else:
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=cfg.label_smoothing)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

# Kiểm tra build nhanh, không train
_tmp_model, _tmp_weights_source = build_classifier_model("densenet121", CFG)
print("Test build DenseNet121 OK. weights_source=", _tmp_weights_source)
del _tmp_model
keras.backend.clear_session()

# ==============================================================================
# 15. Hàm train một model hoàn chỉnh
# ==============================================================================
def get_class_weight_dict(train_df: pd.DataFrame) -> Dict[int, float]:
    y = train_df["diagnosis"].astype(int).values
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def plot_history(history_csv: Path, out_png: Path, title: str) -> None:
    if not history_csv.exists():
        return
    hist = pd.read_csv(history_csv)
    plt.figure(figsize=(8, 4))
    if "loss" in hist.columns:
        plt.plot(hist["loss"], label="train_loss")
    if "val_loss" in hist.columns:
        plt.plot(hist["val_loss"], label="val_loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.show()
    plt.close()


def train_one_model(model_type: str, cfg: TrainConfig, splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    model_type = model_type.lower()
    run_dir = ensure_dir(Path(cfg.output_root) / model_type)
    save_json(asdict(cfg), run_dir / "train_config.json")

    train_seq = DRSequence(splits["train"], cfg, model_type, training=True)
    val_seq = DRSequence(splits["val"], cfg, model_type, training=False)
    test_seq = DRSequence(splits["test"], cfg, model_type, training=False)

    class_weight = get_class_weight_dict(splits["train"]) if (cfg.use_class_weight or cfg.imbalance_strategy == "class_weight") else None
    print("Class weight:", class_weight)
    save_json({"class_weight": class_weight}, run_dir / "class_weight.json")

    model, weights_source = build_classifier_model(model_type, cfg)
    save_json({"weights_source": weights_source}, run_dir / "weights_source.json")

    best_model_path = run_dir / f"{model_type}_best.keras"
    history_csv = run_dir / f"{model_type}_history.csv"

    callbacks = [
        MedicalMetricsCallback(val_seq),
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor=cfg.monitor_metric,
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        EarlyStopping(
            monitor=cfg.monitor_metric,
            mode="max",
            patience=cfg.early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.3,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_lr,
            verbose=1,
        ),
        CSVLogger(str(history_csv), append=False),
    ]

    # Phase 1: train classifier head
    print(f"\n===== {model_type.upper()} | PHASE 1: train head =====")
    # Nếu pretrained có sẵn thì freeze backbone. Nếu weights=None thì không freeze để model học từ đầu.
    if weights_source == "none":
        set_backbone_trainable(model, trainable=True, unfreeze_last_n=None)
    else:
        set_backbone_trainable(model, trainable=False)
    compile_model(model, cfg.head_lr, cfg, train_df=splits["train"])
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=cfg.epochs_head,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2: fine-tune
    print(f"\n===== {model_type.upper()} | PHASE 2: fine-tune =====")
    set_backbone_trainable(model, trainable=True, unfreeze_last_n=cfg.unfreeze_last_n)
    compile_model(model, cfg.finetune_lr, cfg, train_df=splits["train"])
    initial_epoch = cfg.epochs_head
    total_epochs = cfg.epochs_head + cfg.epochs_finetune
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Load best model
    if best_model_path.exists():
        best_model = keras.models.load_model(str(best_model_path), compile=False)
    else:
        best_model = model

    print(f"\n===== {model_type.upper()} | VALIDATION EVALUATION =====")
    y_val, p_val = predict_sequence_proba(best_model, val_seq)
    val_metrics = evaluate_predictions(y_val, p_val, run_dir, f"{model_type}_val")

    print(f"\n===== {model_type.upper()} | TEST EVALUATION =====")
    y_test, p_test = predict_sequence_proba(best_model, test_seq)
    test_metrics = evaluate_predictions(y_test, p_test, run_dir, f"{model_type}_test")

    # Export .h5 cho backend
    h5_path = run_dir / f"{model_type}_best.h5"
    best_model.save(str(h5_path))

    plot_history(history_csv, run_dir / f"{model_type}_loss_curve.png", f"{model_type} loss curve")

    summary = {
        "model_type": model_type,
        "weights_source": weights_source,
        "best_model_path": str(best_model_path),
        "h5_model_path": str(h5_path),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(summary, run_dir / f"{model_type}_summary.json")
    return summary

# ==============================================================================
# 16. Train Model 1 — DenseNet121
# ==============================================================================
# Chạy cell này để train DenseNet121.
# Nếu muốn test nhanh pipeline trước, có thể giảm CFG.epochs_head và CFG.epochs_finetune ở cell cấu hình.

densenet_summary = train_one_model("densenet121", CFG, splits)
print(json.dumps(densenet_summary, indent=2, ensure_ascii=False))

# ==============================================================================
# 17. Train Model 2 — EfficientNetB3
# ==============================================================================
# Chạy cell này để train EfficientNetB3.
# Nếu Kaggle chặn ImageNet weights, code sẽ thử tìm weights offline trong /kaggle/input rồi fallback weights=None.

efficientnet_summary = train_one_model("efficientnetb3", CFG, splits)
print(json.dumps(efficientnet_summary, indent=2, ensure_ascii=False))

# ==============================================================================
# 18. So sánh riêng từng model
# ==============================================================================
def read_model_test_metrics(model_type: str) -> Dict[str, Any]:
    path = Path(CFG.output_root) / model_type / f"{model_type}_test_metrics.json"
    return load_json(path)

rows = []
for mt in ["densenet121", "efficientnetb3"]:
    try:
        m = read_model_test_metrics(mt)
        rows.append({"model": mt, **m})
    except Exception as e:
        print("Skip", mt, e)

comparison_df = pd.DataFrame(rows)
display(comparison_df)
comparison_df.to_csv(Path(CFG.output_root) / "model_comparison_test.csv", index=False)

# ==============================================================================
# 19. Ensemble: weighted average + threshold optimization
# ==============================================================================
def load_predictions_csv(model_type: str, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(CFG.output_root) / model_type / f"{model_type}_{split_name}_predictions.csv"
    pred = pd.read_csv(path)
    y_true = pred["y_true"].values.astype(np.int32)
    probs = pred[[f"prob_{c}" for c in range(NUM_CLASSES)]].values.astype(np.float32)
    return y_true, probs


def expected_severity_score(probs: np.ndarray) -> np.ndarray:
    classes = np.arange(NUM_CLASSES, dtype=np.float32)
    return np.sum(probs * classes.reshape(1, -1), axis=1)


def apply_thresholds(scores: np.ndarray, thresholds: List[float]) -> np.ndarray:
    thresholds = sorted([float(t) for t in thresholds])
    pred = np.zeros_like(scores, dtype=np.int32)
    pred[scores >= thresholds[0]] = 1
    pred[scores >= thresholds[1]] = 2
    pred[scores >= thresholds[2]] = 3
    pred[scores >= thresholds[3]] = 4
    return pred


def optimize_thresholds_grid(y_true: np.ndarray, scores: np.ndarray) -> Tuple[List[float], float]:
    """Grid search đơn giản, ổn định cho 5 lớp."""
    best_qwk = -999
    best_t = [0.5, 1.5, 2.5, 3.5]

    # Tìm quanh ngưỡng mặc định, bước 0.1 để không quá lâu
    t1_values = np.arange(0.3, 1.3, 0.1)
    t2_values = np.arange(1.0, 2.2, 0.1)
    t3_values = np.arange(1.8, 3.2, 0.1)
    t4_values = np.arange(2.6, 4.2, 0.1)

    for t1 in t1_values:
        for t2 in t2_values:
            if t2 <= t1:
                continue
            for t3 in t3_values:
                if t3 <= t2:
                    continue
                for t4 in t4_values:
                    if t4 <= t3:
                        continue
                    pred = apply_thresholds(scores, [t1, t2, t3, t4])
                    qwk = cohen_kappa_score(y_true, pred, weights="quadratic")
                    if qwk > best_qwk:
                        best_qwk = qwk
                        best_t = [float(t1), float(t2), float(t3), float(t4)]
    return best_t, float(best_qwk)


def evaluate_ensemble() -> Dict[str, Any]:
    ens_dir = ensure_dir(Path(CFG.output_root) / "ensemble")

    y_val_d, p_val_d = load_predictions_csv("densenet121", "val")
    y_val_e, p_val_e = load_predictions_csv("efficientnetb3", "val")
    y_test_d, p_test_d = load_predictions_csv("densenet121", "test")
    y_test_e, p_test_e = load_predictions_csv("efficientnetb3", "test")

    assert np.array_equal(y_val_d, y_val_e), "Validation labels mismatch"
    assert np.array_equal(y_test_d, y_test_e), "Test labels mismatch"
    y_val = y_val_d
    y_test = y_test_d

    weight_rows = []
    best = {"weight_efficientnetb3": None, "val_qwk": -999}
    for w in np.arange(0.0, 1.01, 0.05):
        p_val = w * p_val_e + (1 - w) * p_val_d
        pred_val = np.argmax(p_val, axis=1)
        acc = accuracy_score(y_val, pred_val)
        qwk = cohen_kappa_score(y_val, pred_val, weights="quadratic")
        mf1 = f1_score(y_val, pred_val, average="macro", zero_division=0)
        row = {"weight_efficientnetb3": float(w), "weight_densenet121": float(1 - w), "val_accuracy": float(acc), "val_qwk": float(qwk), "val_macro_f1": float(mf1)}
        weight_rows.append(row)
        if qwk > best["val_qwk"]:
            best = row.copy()

    weight_df = pd.DataFrame(weight_rows)
    display(weight_df.sort_values("val_qwk", ascending=False).head(10))
    weight_df.to_csv(ens_dir / "ensemble_weight_search.csv", index=False)

    w = best["weight_efficientnetb3"]
    p_val_best = w * p_val_e + (1 - w) * p_val_d
    p_test_best = w * p_test_e + (1 - w) * p_test_d

    # Đánh giá argmax
    print("\nENSEMBLE ARGMAX - VALIDATION")
    val_metrics_argmax = evaluate_predictions(y_val, p_val_best, ens_dir, "ensemble_val_argmax")
    print("\nENSEMBLE ARGMAX - TEST")
    test_metrics_argmax = evaluate_predictions(y_test, p_test_best, ens_dir, "ensemble_test_argmax")

    # Threshold optimization trên validation, áp dụng cho test
    val_scores = expected_severity_score(p_val_best)
    thresholds, threshold_val_qwk = optimize_thresholds_grid(y_val, val_scores)
    print("Best thresholds:", thresholds, "val_qwk:", threshold_val_qwk)

    test_scores = expected_severity_score(p_test_best)
    test_pred_thr = apply_thresholds(test_scores, thresholds)

    # Lưu predictions threshold
    thr_pred_df = pd.DataFrame({"y_true": y_test, "y_pred": test_pred_thr, "expected_score": test_scores})
    for c in range(NUM_CLASSES):
        thr_pred_df[f"prob_{c}"] = p_test_best[:, c]
    thr_pred_df.to_csv(ens_dir / "ensemble_test_threshold_predictions.csv", index=False)

    thr_metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred_thr)),
        "qwk": float(cohen_kappa_score(y_test, test_pred_thr, weights="quadratic")),
        "macro_f1": float(f1_score(y_test, test_pred_thr, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_test, test_pred_thr, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, test_pred_thr, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_test, test_pred_thr)),
        "thresholds": thresholds,
        "threshold_val_qwk": threshold_val_qwk,
    }
    save_json(thr_metrics, ens_dir / "ensemble_test_threshold_metrics.json")
    print("\nENSEMBLE THRESHOLD - TEST")
    print(json.dumps(thr_metrics, indent=2, ensure_ascii=False))

    cm = confusion_matrix(y_test, test_pred_thr, labels=list(range(NUM_CLASSES)))
    save_confusion_matrix_png(cm, ens_dir / "ensemble_test_threshold_confusion_matrix.png", "ensemble threshold confusion matrix")

    results = {
        "best_weight": best,
        "val_metrics_argmax": val_metrics_argmax,
        "test_metrics_argmax": test_metrics_argmax,
        "threshold_metrics": thr_metrics,
        "class_names": CLASS_NAMES,
        "img_size": CFG.img_size,
        "preprocessing": "crop_black_borders + CLAHE_LAB + resize; no_ESRGAN",
        "imbalance_strategy": CFG.imbalance_strategy,
        "use_focal_loss": CFG.use_focal_loss,
    }
    save_json(results, ens_dir / "ensemble_results.json")
    return results

ensemble_results = evaluate_ensemble()
print(json.dumps(ensemble_results, indent=2, ensure_ascii=False))

# ==============================================================================
# 20. Grad-CAM kiểm tra trực quan tùy chọn
# ==============================================================================
# Cell tùy chọn: tạo Grad-CAM sau khi train để minh họa trong báo cáo.
# Nếu chưa cần, có thể bỏ qua cell này.

def find_last_conv_layer(model: keras.Model) -> Optional[str]:
    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model):
            for sub in reversed(layer.layers):
                if len(getattr(sub, "output_shape", [])) == 4 or "conv" in sub.name.lower():
                    try:
                        if len(sub.output.shape) == 4:
                            return f"{layer.name}/{sub.name}"
                    except Exception:
                        pass
        else:
            try:
                if len(layer.output.shape) == 4:
                    return layer.name
            except Exception:
                pass
    return None

print("Grad-CAM cell available. Use after train if needed.")

# ==============================================================================
# 21. Export artifact cho backend Flask
# ==============================================================================
def copy_if_exists(src: str | Path, dst: str | Path) -> bool:
    src, dst = Path(src), Path(dst)
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        print("Copied", src, "->", dst)
        return True
    print("Missing", src)
    return False

EXPORT_DIR = ensure_dir(ARTIFACTS_DIR)

# Copy 2 model .h5
copy_if_exists(Path(CFG.output_root) / "densenet121" / "densenet121_best.h5", EXPORT_DIR / "densenet121_best.h5")
copy_if_exists(Path(CFG.output_root) / "efficientnetb3" / "efficientnetb3_best.h5", EXPORT_DIR / "efficientnetb3_best.h5")

# Copy ensemble config
copy_if_exists(Path(CFG.output_root) / "ensemble" / "ensemble_results.json", EXPORT_DIR / "ensemble_results.json")
copy_if_exists(Path(CFG.output_root) / "ensemble" / "ensemble_test_threshold_metrics.json", EXPORT_DIR / "threshold_results.json")

# Metadata cho backend
metadata = {
    "models": ["densenet121", "efficientnetb3"],
    "img_size": CFG.img_size,
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "preprocessing": {
        "crop_black_borders": True,
        "clahe_lab": CFG.use_clahe,
        "esrgan": False,
        "resize": [CFG.img_size, CFG.img_size],
    },
    "imbalance_strategy": CFG.imbalance_strategy,
    "use_focal_loss": CFG.use_focal_loss,
    "notes": "DenseNet121 + EfficientNetB3. Imbalance handling v3. No ESRGAN.",
}
save_json(metadata, EXPORT_DIR / "model_metadata.json")

# Zip backend artifacts
backend_zip = Path("/kaggle/working/backend_model_artifacts.zip")
if backend_zip.exists():
    backend_zip.unlink()
shutil.make_archive(str(backend_zip).replace(".zip", ""), "zip", EXPORT_DIR)
print("Backend artifact zip:", backend_zip)

# Zip toàn bộ training outputs để đưa vào luận văn
outputs_zip = Path("/kaggle/working/training_outputs_bundle.zip")
if outputs_zip.exists():
    outputs_zip.unlink()
shutil.make_archive(str(outputs_zip).replace(".zip", ""), "zip", Path(CFG.output_root))
print("Training outputs zip:", outputs_zip)

print("\nDownload these files from Kaggle Output:")
print("-", backend_zip)
print("-", outputs_zip)

# ==============================================================================
# 22. Checklist sau train
# ==============================================================================
