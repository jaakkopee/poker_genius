"""Small PyTorch models for card rank/suit symbol classification.

This module provides:
- Synthetic bootstrap dataset generation for initial training.
- Lightweight CNN classifier for rank and suit symbols.
- Prediction helpers used by the live recognition pipeline.
- Incremental fine-tuning from user-labeled samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    Dataset = object

RANK_CLASSES = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUIT_CLASSES = ["c", "d", "h", "s"]
SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}

MODEL_DIR = Path("models")
DATA_DIR = Path("training_data")
RANK_MODEL_PATH = MODEL_DIR / "rank_symbol_cnn.pt"
SUIT_MODEL_PATH = MODEL_DIR / "suit_symbol_cnn.pt"
BOOTSTRAP_DIR = DATA_DIR / "bootstrap"
USER_LABELS_DIR = DATA_DIR / "user_labeled"

FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
)

Patch = Tuple[Image.Image, Image.Image]


class SymbolCNN(nn.Module):
    """Very small CNN for 64x64 grayscale symbol patches."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class SymbolDataset(Dataset):
    """Directory dataset where classes are subfolders and files are PNG/JPG."""

    def __init__(self, folders: List[Path], classes: List[str]):
        self.classes = classes
        self.class_to_index = {name: idx for idx, name in enumerate(classes)}
        self.samples: List[Tuple[Path, int]] = []
        for root in folders:
            if not root.exists():
                continue
            for class_name in classes:
                class_dir = root / class_name
                if not class_dir.exists():
                    continue
                for path in sorted(class_dir.glob("*")):
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                        self.samples.append((path, self.class_to_index[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        arr = np.array(Image.open(path).convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)
        arr = (arr / 255.0 - 0.5) / 0.5
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor, label


@dataclass
class TrainResult:
    samples: int
    epochs: int
    final_loss: float
    model_path: Path


def has_torch() -> bool:
    return torch is not None


@lru_cache(maxsize=1)
def _load_fonts(size: int) -> Tuple[ImageFont.FreeTypeFont, ...]:
    fonts = []
    for path in FONT_CANDIDATES:
        try:
            fonts.append(ImageFont.truetype(path, size))
        except OSError:
            continue
    if not fonts:
        fonts.append(ImageFont.load_default())
    return tuple(fonts)


def extract_rank_and_suit_crops(card_image: Image.Image) -> Patch:
    """Extract crops aligned with the existing recognizer's top-left symbol logic."""
    oriented = card_image
    if oriented.width > oriented.height:
        oriented = oriented.rotate(90, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    oriented = oriented.resize((180, 252), Image.Resampling.LANCZOS)
    rank_crop = oriented.crop((0, 0, int(oriented.width * 0.50), int(oriented.height * 0.40)))
    suit_crop = oriented.crop((0, int(oriented.height * 0.15), int(oriented.width * 0.50), int(oriented.height * 0.55)))
    return rank_crop, suit_crop


def _preprocess_symbol_patch(pil_image: Image.Image) -> Image.Image:
    gray = np.array(pil_image.convert("L"))
    blur = cv2_gaussian(gray)
    binary = cv2_otsu_binary_inv(blur)
    points = np.argwhere(binary > 0)
    if points.size == 0:
        return Image.fromarray(np.zeros((64, 64), dtype=np.uint8))

    y_min, x_min = points.min(axis=0)
    y_max, x_max = points.max(axis=0)
    crop = binary[y_min : y_max + 1, x_min : x_max + 1]

    pad = max(4, int(max(crop.shape) * 0.15))
    padded = np.pad(crop, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    img = Image.fromarray(padded).resize((64, 64), Image.Resampling.BILINEAR)
    return img


def cv2_gaussian(gray: np.ndarray) -> np.ndarray:
    # Small fallback blur implementation that avoids hard dependency on cv2 in this module.
    pil = Image.fromarray(gray)
    return np.array(pil.filter(ImageFilter.GaussianBlur(radius=1.0)))


def cv2_otsu_binary_inv(gray: np.ndarray) -> np.ndarray:
    threshold = otsu_threshold(gray)
    binary = (gray < threshold).astype(np.uint8) * 255
    return binary


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold = 127

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * ((m_b - m_f) ** 2)
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return int(threshold)


def preprocess_for_model(pil_image: Image.Image) -> torch.Tensor:
    patch = _preprocess_symbol_patch(pil_image)
    arr = np.array(patch, dtype=np.float32)
    arr = (arr / 255.0 - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def _render_symbol(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    canvas = Image.new("L", (128, 128), 255)
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(
        ((canvas.width - text_w) / 2 - bbox[0], (canvas.height - text_h) / 2 - bbox[1]),
        text,
        font=font,
        fill=0,
    )
    return canvas


def _augment_symbol(base: Image.Image) -> Image.Image:
    angle = random.uniform(-10, 10)
    scale = random.uniform(0.8, 1.2)
    translate_x = random.randint(-6, 6)
    translate_y = random.randint(-6, 6)

    rot = base.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    w, h = rot.size
    resized = rot.resize((max(16, int(w * scale)), max(16, int(h * scale))), Image.Resampling.BILINEAR)

    canvas = Image.new("L", (128, 128), 255)
    x = (128 - resized.width) // 2 + translate_x
    y = (128 - resized.height) // 2 + translate_y
    canvas.paste(resized, (x, y))

    if random.random() < 0.4:
        canvas = ImageOps.autocontrast(canvas)
    if random.random() < 0.35:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))

    # Add light noise.
    arr = np.array(canvas, dtype=np.int16)
    noise = np.random.normal(0, 8, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def ensure_bootstrap_dataset(samples_per_class: int = 120) -> None:
    """Create a synthetic baseline dataset if it doesn't exist yet."""
    rank_root = BOOTSTRAP_DIR / "rank"
    suit_root = BOOTSTRAP_DIR / "suit"

    if rank_root.exists() and suit_root.exists():
        return

    for root in (rank_root, suit_root):
        root.mkdir(parents=True, exist_ok=True)

    for rank in RANK_CLASSES:
        class_dir = rank_root / rank
        class_dir.mkdir(parents=True, exist_ok=True)
        glyphs = [rank] if rank != "T" else ["10", "T"]
        for idx in range(samples_per_class):
            glyph = random.choice(glyphs)
            font = random.choice(_load_fonts(size=random.randint(56, 78)))
            base = _render_symbol(glyph, font)
            aug = _augment_symbol(base)
            patch = _preprocess_symbol_patch(aug)
            patch.save(class_dir / f"{rank}_{idx:04d}.png")

    for suit in SUIT_CLASSES:
        class_dir = suit_root / suit
        class_dir.mkdir(parents=True, exist_ok=True)
        glyphs = [SUIT_SYMBOLS[suit], suit.upper()]
        for idx in range(samples_per_class):
            glyph = random.choice(glyphs)
            font = random.choice(_load_fonts(size=random.randint(54, 74)))
            base = _render_symbol(glyph, font)
            aug = _augment_symbol(base)
            patch = _preprocess_symbol_patch(aug)
            patch.save(class_dir / f"{suit}_{idx:04d}.png")


def _train_model(
    dataset: SymbolDataset,
    classes: List[str],
    out_path: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    progress: Optional[Callable[[str], None]] = None,
) -> TrainResult:
    if not has_torch():
        raise RuntimeError("PyTorch is not available. Install with: pip install torch")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(dataset) == 0:
        raise RuntimeError("No training samples found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SymbolCNN(num_classes=len(classes)).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    final_loss = 0.0
    model.train()
    for epoch in range(epochs):
        running = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * batch_x.size(0)
            seen += batch_x.size(0)

        final_loss = running / max(1, seen)
        if progress:
            progress(f"epoch {epoch + 1}/{epochs}: loss={final_loss:.4f}")

    payload = {
        "state_dict": model.state_dict(),
        "classes": classes,
        "created_at": time.time(),
    }
    torch.save(payload, out_path)
    _load_model_cached.cache_clear()
    return TrainResult(samples=len(dataset), epochs=epochs, final_loss=final_loss, model_path=out_path)


def train_rank_and_suit_models(
    include_user_data: bool = True,
    bootstrap_if_missing: bool = True,
    epochs: int = 8,
    lr: float = 1e-3,
    batch_size: int = 64,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, TrainResult]:
    """Train both rank and suit models from bootstrap + optional user data."""
    if bootstrap_if_missing:
        ensure_bootstrap_dataset()

    rank_dirs = [BOOTSTRAP_DIR / "rank"]
    suit_dirs = [BOOTSTRAP_DIR / "suit"]
    if include_user_data:
        rank_dirs.append(USER_LABELS_DIR / "rank")
        suit_dirs.append(USER_LABELS_DIR / "suit")

    rank_set = SymbolDataset(rank_dirs, RANK_CLASSES)
    suit_set = SymbolDataset(suit_dirs, SUIT_CLASSES)

    if progress:
        progress(f"rank samples: {len(rank_set)}, suit samples: {len(suit_set)}")

    rank_res = _train_model(rank_set, RANK_CLASSES, RANK_MODEL_PATH, epochs, lr, batch_size, progress)
    suit_res = _train_model(suit_set, SUIT_CLASSES, SUIT_MODEL_PATH, epochs, lr, batch_size, progress)
    return {"rank": rank_res, "suit": suit_res}


def add_labeled_card_sample(card_image: Image.Image, card: str, source_tag: str = "capture") -> Tuple[Path, Path]:
    """Save one user-labeled card as rank and suit symbol samples."""
    if len(card) != 2:
        raise ValueError(f"Invalid card label: {card}")

    rank = card[0].upper()
    suit = card[1].lower()
    if rank not in RANK_CLASSES or suit not in SUIT_CLASSES:
        raise ValueError(f"Unsupported card label: {card}")

    rank_crop, suit_crop = extract_rank_and_suit_crops(card_image)
    rank_patch = _preprocess_symbol_patch(rank_crop)
    suit_patch = _preprocess_symbol_patch(suit_crop)

    timestamp = int(time.time() * 1000)
    rank_dir = USER_LABELS_DIR / "rank" / rank
    suit_dir = USER_LABELS_DIR / "suit" / suit
    rank_dir.mkdir(parents=True, exist_ok=True)
    suit_dir.mkdir(parents=True, exist_ok=True)

    rank_path = rank_dir / f"{source_tag}_{timestamp}_{rank}.png"
    suit_path = suit_dir / f"{source_tag}_{timestamp}_{suit}.png"
    rank_patch.save(rank_path)
    suit_patch.save(suit_path)
    return rank_path, suit_path


@lru_cache(maxsize=2)
def _load_model_cached(model_path: str):
    if not has_torch():
        return None, []
    path = Path(model_path)
    if not path.exists():
        return None, []

    payload = torch.load(path, map_location="cpu")
    classes = payload.get("classes", [])
    model = SymbolCNN(num_classes=len(classes))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, classes


def predict_symbol(pil_image: Image.Image, model_path: Path) -> Tuple[Optional[str], float]:
    """Predict symbol class with confidence. Returns (label, confidence)."""
    if not has_torch():
        return None, 0.0
    model, classes = _load_model_cached(str(model_path))
    if model is None or not classes:
        return None, 0.0

    with torch.no_grad():
        inp = preprocess_for_model(pil_image)
        logits = model(inp)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, idx = torch.max(probs, dim=0)
        return classes[int(idx)], float(conf.item())


def predict_rank_symbol(pil_image: Image.Image) -> Tuple[Optional[str], float]:
    return predict_symbol(pil_image, RANK_MODEL_PATH)


def predict_suit_symbol(pil_image: Image.Image) -> Tuple[Optional[str], float]:
    return predict_symbol(pil_image, SUIT_MODEL_PATH)


def model_status() -> Dict[str, bool]:
    return {
        "torch_available": has_torch(),
        "rank_model": RANK_MODEL_PATH.exists(),
        "suit_model": SUIT_MODEL_PATH.exists(),
    }
