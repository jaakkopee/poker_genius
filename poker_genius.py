"""
Poker Genius - Screen capture + OCR + GTO poker advisor
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import re
import sys
from functools import lru_cache
from itertools import combinations
from typing import Optional, List, Tuple

import pytesseract
try:
    import easyocr
except Exception:
    easyocr = None
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageGrab, ImageTk
import cv2
import numpy as np


# ──────────────────────────────────────────────
# Card recognition helpers
# ──────────────────────────────────────────────

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["c", "d", "h", "s"]  # clubs, diamonds, hearts, spades

RANK_ALIASES = {
    "10": "T", "1O": "T", "IO": "T",
    "J": "J", "Q": "Q", "K": "K", "A": "A",
    "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9",
}

SUIT_ALIASES = {
    "c": "c", "♣": "c", "clubs": "c",
    "d": "d", "♦": "d", "diamonds": "d",
    "h": "h", "♥": "h", "hearts": "h",
    "s": "s", "♠": "s", "spades": "s",
}

RANK_VALUES = {r: i for i, r in enumerate(RANKS)}
OCR_ROTATION_ANGLES = (-5, 0, 5)
OCR_CONFIG = "--psm 11"
OCR_ENGINES = ("pytesseract", "easyocr")
CARD_ORIENTATIONS = (0, 180)
TEMPLATE_CONFIDENCE_THRESHOLD = 0.50
CARD_CANVAS = (180, 252)
SYMBOL_TEMPLATE_SIZE = (64, 64)
# Very permissive HSV mask - accept darker cards (V >= 50 instead of 100)
CARD_MASK_LOW = np.array([0, 0, 80], dtype=np.uint8)
CARD_MASK_HIGH = np.array([180, 90, 255], dtype=np.uint8)
SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
RED_SUITS = {"d", "h"}
BLACK_SUITS = {"c", "s"}
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
)


def preprocess_for_ocr(pil_image: Image.Image, upscale_factor: int = 2) -> Image.Image:
    """Enhance image for better OCR accuracy on card text."""
    gray = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    thresholded = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    processed = Image.fromarray(cv2.medianBlur(thresholded, 3))
    processed = processed.filter(ImageFilter.SHARPEN)
    processed = ImageEnhance.Contrast(processed).enhance(2.0)
    return processed.resize((processed.width * upscale_factor, processed.height * upscale_factor), Image.LANCZOS)


def order_points(points: np.ndarray) -> np.ndarray:
    """Return rectangle corners ordered as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    diffs = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def normalize_symbol_patch(pil_image: Image.Image, size: Tuple[int, int] = SYMBOL_TEMPLATE_SIZE) -> Optional[np.ndarray]:
    """Convert a symbol crop into a centered binary patch for OCR/template matching."""
    gray = np.array(pil_image.convert("L"))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    if points is None:
        return None

    x, y, width, height = cv2.boundingRect(points)
    if width < 2 or height < 2:
        return None

    cropped = binary[y:y + height, x:x + width]
    pad = max(4, int(max(width, height) * 0.15))
    cropped = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=8)
def load_template_fonts(size: int) -> Tuple[ImageFont.FreeTypeFont, ...]:
    """Load a small set of fonts that can render card ranks and suit symbols."""
    fonts = []
    for path in FONT_CANDIDATES:
        try:
            fonts.append(ImageFont.truetype(path, size))
        except OSError:
            continue

    if not fonts:
        fonts.append(ImageFont.load_default())
    return tuple(fonts)


def render_template_variants(symbol: str, font_size: int) -> List[np.ndarray]:
    """Render one or more binary templates for a rank or suit symbol."""
    templates = []
    for font in load_template_fonts(font_size):
        canvas = Image.new("L", (128, 128), 255)
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), symbol, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            ((canvas.width - text_w) / 2 - bbox[0], (canvas.height - text_h) / 2 - bbox[1]),
            symbol,
            font=font,
            fill=0,
        )
        normalized = normalize_symbol_patch(canvas)
        if normalized is not None:
            templates.append(normalized)
    return templates


@lru_cache(maxsize=1)
def get_rank_templates() -> dict:
    """Build canonical rank templates, including a dedicated 10 glyph variant."""
    variants = {rank: [rank] for rank in RANKS}
    variants["T"] = ["10", "T"]
    return {
        rank: [template for text in texts for template in render_template_variants(text, 72)]
        for rank, texts in variants.items()
    }


@lru_cache(maxsize=1)
def get_suit_templates() -> dict:
    """Build suit templates using both suit symbols and fallback letters."""
    return {
        suit: [
            template
            for text in (SUIT_SYMBOLS[suit], suit.upper())
            for template in render_template_variants(text, 66)
        ]
        for suit in SUITS
    }


def template_match_symbol(pil_image: Image.Image, templates: dict, allowed: Optional[set] = None, debug_label: str = "") -> Tuple[Optional[str], float]:
    """Return the best matching canonical symbol and its normalized correlation score."""
    normalized = normalize_symbol_patch(pil_image)
    if normalized is None:
        return None, 0.0

    best_symbol = None
    best_score = 0.0
    all_scores = {}  # Track all symbol scores for debugging
    candidate = normalized.astype(np.float32)
    for symbol, variants in templates.items():
        if allowed and symbol not in allowed:
            continue
        max_score_for_symbol = 0.0
        for template in variants:
            score = float(cv2.matchTemplate(candidate, template.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0][0])
            if score > max_score_for_symbol:
                max_score_for_symbol = score
        all_scores[symbol] = max_score_for_symbol
        if max_score_for_symbol > best_score:
            best_symbol = symbol
            best_score = max_score_for_symbol
    
    # Debug: show top 3 matches
    if debug_label and all_scores:
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        debug_str = " | ".join([f"{sym}:{sc:.3f}" for sym, sc in sorted_scores])
        import sys
        print(f"    [{debug_label}] {debug_str}", file=sys.stderr, flush=True)

    return best_symbol, best_score


def ocr_single_symbol(pil_image: Image.Image, whitelist: str) -> str:
    """Run Tesseract on a tight symbol crop."""
    normalized = normalize_symbol_patch(pil_image)
    if normalized is None:
        return ""
    ocr_image = Image.fromarray(255 - normalized)
    return pytesseract.image_to_string(
        ocr_image,
        config=f"--psm 10 -c tessedit_char_whitelist={whitelist}",
    ).strip()


@lru_cache(maxsize=1)
def get_easyocr_reader():
    """Lazily initialize EasyOCR reader (English only for card glyphs)."""
    if easyocr is None:
        raise RuntimeError("EasyOCR is not installed. Install it with: pip install easyocr")
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def ocr_text_with_engine(pil_image: Image.Image, engine: str = "pytesseract", psm: int = 11,
                         whitelist: str = "") -> str:
    """Run OCR with selected engine; supports pytesseract and easyocr."""
    engine = (engine or "pytesseract").lower()
    if engine == "easyocr":
        reader = get_easyocr_reader()
        arr = np.array(pil_image.convert("RGB"))
        kwargs = {"detail": 0, "paragraph": False}
        if whitelist:
            kwargs["allowlist"] = whitelist
        results = reader.readtext(arr, **kwargs)
        return " ".join(results).strip()

    config = f"--psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(pil_image, config=config).strip()


def ocr_single_symbol_with_engine(pil_image: Image.Image, whitelist: str, engine: str = "pytesseract") -> str:
    """Run OCR on a tight symbol crop with selected engine."""
    normalized = normalize_symbol_patch(pil_image)
    if normalized is None:
        return ""
    ocr_image = Image.fromarray(255 - normalized)
    return ocr_text_with_engine(ocr_image, engine=engine, psm=10, whitelist=whitelist)


def normalize_rank_symbol(text: str) -> Optional[str]:
    """Normalize OCR/template output to a canonical rank."""
    cleaned = re.sub(r"[^0-9TJQKAIO]", "", text.upper())
    if not cleaned:
        return None
    if "10" in cleaned or "TO" in cleaned or cleaned.startswith("T"):
        return "T"
    return RANK_ALIASES.get(cleaned[0])


def normalize_suit_symbol(text: str) -> Optional[str]:
    """Normalize OCR/template output to a canonical suit."""
    for token in text:
        suit = SUIT_ALIASES.get(token.lower())
        if suit:
            return suit
    return None


def estimate_red_suit(pil_image: Image.Image, red_ratio_threshold: float = 0.38) -> bool:
    """Estimate whether the symbol is red or black from the original crop."""
    rgb = np.array(pil_image.convert("RGB"))
    
    # Find foreground pixels directly from the original image
    # (assuming symbol is darker than white background)
    gray = np.array(pil_image.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = binary == 0
    
    if not np.any(mask):
        return False
    
    foreground = rgb[mask]
    
    # Calculate average RGB values
    avg_r = foreground[:, 0].mean()
    avg_g = foreground[:, 1].mean()
    avg_b = foreground[:, 2].mean()
    
    # Check for red dominance (red channel significantly higher than green and blue)
    # Red suits have R > G and R > B
    red_score = (avg_r - avg_g) + (avg_r - avg_b)
    
    # If red channel is clearly dominant, it's a red suit
    if red_score > 20:  # Red is at least 10 higher than both G and B on average
        return True
    
    # Check overall darkness - pure black suits will have very low values across all channels
    avg_intensity = foreground.mean()
    if avg_intensity < 50:  # Very dark → black suit
        return False
    
    # For medium-dark colors, check red ratio
    red_ratio = avg_r / (avg_r + avg_g + avg_b + 1e-6)
    return red_ratio > red_ratio_threshold


def rectify_card_region(rgb_image: np.ndarray, rect: Tuple[Tuple[float, float], Tuple[float, float], float]) -> Image.Image:
    """Perspective-warp a candidate contour into a standard portrait card canvas."""
    source = order_points(cv2.boxPoints(rect).astype("float32"))
    destination = np.array(
        [
            [0, 0],
            [CARD_CANVAS[0] - 1, 0],
            [CARD_CANVAS[0] - 1, CARD_CANVAS[1] - 1],
            [0, CARD_CANVAS[1] - 1],
        ],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(rgb_image, transform, CARD_CANVAS)
    return Image.fromarray(warped)


def find_card_regions(pil_image: Image.Image, white_ratio_threshold: float = 0.24,
                      aspect_ratio_min: float = 0.45, aspect_ratio_max: float = 0.88,
                      fill_ratio_threshold: float = 0.38,
                      hsv_v_min: int = 95, hsv_v_max: int = 255,
                      hsv_s_max: int = 75) -> List[Tuple[Tuple[int, int], Image.Image, Tuple[int, int, int, int], float]]:
    """Locate bright, card-shaped regions and rectify them into portrait card crops. Returns (center, warped_image, bbox, angle)."""
    rgb = np.array(pil_image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    
    # Create HSV mask thresholds dynamically
    card_mask_low = np.array([0, 0, hsv_v_min], dtype=np.uint8)
    card_mask_high = np.array([180, hsv_s_max, hsv_v_max], dtype=np.uint8)
    
    # DEBUG: Analyze HSV values to understand why mask might be empty
    import sys
    h_min, h_max = hsv[:,:,0].min(), hsv[:,:,0].max()
    s_min, s_max = hsv[:,:,1].min(), hsv[:,:,1].max()
    v_min, v_max = hsv[:,:,2].min(), hsv[:,:,2].max()
    print(f"[HSV-RANGE] H: {h_min}-{h_max}, S: {s_min}-{s_max}, V: {v_min}-{v_max}", file=sys.stderr, flush=True)
    print(f"[HSV-MASK] Looking for: H:{card_mask_low[0]}-{card_mask_high[0]}, S:{card_mask_low[1]}-{card_mask_high[1]}, V:{card_mask_low[2]}-{card_mask_high[2]}", file=sys.stderr, flush=True)
    
    mask = cv2.inRange(hsv, card_mask_low, card_mask_high)
    
    # DEBUG: Check how much of the image passed the mask
    mask_coverage = (mask > 0).sum() / mask.size
    print(f"[HSV-MASK] Coverage: {mask_coverage*100:.2f}% of image passed HSV filter", file=sys.stderr, flush=True)
    if mask_coverage > 0.60:
        print("[HSV-MASK] Coverage is too high; try lowering HSV S max or raising HSV V min.", file=sys.stderr, flush=True)
    
    # DEBUG: Save raw mask before morphological operations
    import os
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_crops")
    os.makedirs(debug_dir, exist_ok=True)
    Image.fromarray(mask).save(f"{debug_dir}/01a_hsv_mask_raw.png")
    
    close_kernel = np.ones((3, 3), np.uint8)
    open_kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    
    # DEBUG: Save mask and input image
    import os
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_crops")
    os.makedirs(debug_dir, exist_ok=True)
    Image.fromarray(rgb).save(f"{debug_dir}/00_input_image.png")
    Image.fromarray(mask).save(f"{debug_dir}/01_hsv_mask_processed.png")

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = rgb.shape[0] * rgb.shape[1]
    min_area = max(2500, image_area * 0.001)
    max_area = image_area * 0.08
    candidates = []
    
    # Debug counters
    rejected_by_area = 0
    rejected_by_size = 0
    rejected_by_ratio = 0
    rejected_by_white = 0
    print(f"[REGION-DETECT] Found {len(contours)} contours, white_ratio_threshold={white_ratio_threshold:.2f}", file=sys.stderr, flush=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            rejected_by_area += 1
            continue

        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (width, height), angle_deg = rect
        is_bottom_zone = center_y > rgb.shape[0] * 0.58

        # Enforce stronger geometry gates to filter out isolated suit/rank symbols.
        min_short_side = 55 if is_bottom_zone else 70
        if min(width, height) < min_short_side:
            rejected_by_size += 1
            continue

        local_min_area = max(3000, min_area * 1.0) if is_bottom_zone else max(6000, min_area * 1.0)
        if area < local_min_area:
            rejected_by_area += 1
            continue

        # Hole cards can be partially clipped/covered near the bottom of the table view.
        # In that zone, allow looser shape/coverage checks to keep those candidates.

        short_side, long_side = sorted((width, height))
        ratio = short_side / long_side
        fill_ratio = area / max(width * height, 1)

        ratio_min = aspect_ratio_min
        ratio_max = aspect_ratio_max
        min_fill = fill_ratio_threshold
        if is_bottom_zone:
            ratio_min = max(0.35, aspect_ratio_min - 0.18)
            ratio_max = min(0.95, aspect_ratio_max + 0.18)
            min_fill = max(0.20, fill_ratio_threshold * 0.55)

        if not (ratio_min <= ratio <= ratio_max and fill_ratio >= min_fill):
            rejected_by_ratio += 1
            continue

        x, y, box_w, box_h = cv2.boundingRect(contour)
        warped = rectify_card_region(rgb, rect)
        warped_hsv = cv2.cvtColor(np.array(warped.convert("RGB")), cv2.COLOR_RGB2HSV)
        white_ratio = float((cv2.inRange(warped_hsv, card_mask_low, card_mask_high) > 0).mean())
        # Check if enough of the card is bright/white to be valid
        min_white_ratio = white_ratio_threshold * (0.55 if is_bottom_zone else 1.0)
        if white_ratio < min_white_ratio:
            rejected_by_white += 1
            print(f"[REGION-REJECT] Rejected region at ({int(center_x)},{int(center_y)}): white_ratio={white_ratio:.2f} < threshold={min_white_ratio:.2f}", file=sys.stderr, flush=True)
            continue
        
        # DEBUG: Save accepted warped regions
        import os
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_crops")
        os.makedirs(debug_dir, exist_ok=True)
        region_num = len(candidates) + 1
        warped.save(f"{debug_dir}/region_{region_num}_warped_at_{int(center_x)}_{int(center_y)}.png")
        print(f"[REGION-ACCEPT] Region #{region_num} at ({int(center_x)},{int(center_y)}): white_ratio={white_ratio:.2f}, angle={angle_deg:.1f}°, saved to debug_crops/", file=sys.stderr, flush=True)
        
        # Normalize angle: cv2.minAreaRect gives -90 to 0 range, convert to our coordinate system
        normalized_angle = -angle_deg if width > height else -(90 + angle_deg)
        candidates.append(((int(center_x), int(center_y)), warped, (x, y, box_w, box_h), area, normalized_angle))

    candidates.sort(key=lambda item: item[3], reverse=True)
    deduped = []
    for center, warped, bbox, area, angle in candidates:
        if any(abs(center[0] - prev_center[0]) < 100 and abs(center[1] - prev_center[1]) < 100 for prev_center, _, _, _, _ in deduped):
            continue
        deduped.append((center, warped, bbox, area, angle))

    deduped.sort(key=lambda item: (item[0][1], item[0][0]))
    
    # Debug summary
    print(f"[REGION-SUMMARY] Rejected: area={rejected_by_area}, size={rejected_by_size}, ratio={rejected_by_ratio}, white={rejected_by_white} | Accepted: {len(deduped)}", file=sys.stderr, flush=True)
    
    # If HSV detection found very few cards, try edge-based detection as fallback
    if len(deduped) < 3:
        print(f"[HSV-FALLBACK] Only found {len(deduped)} cards with HSV, trying edge detection...", file=sys.stderr, flush=True)
        edge_cards = find_card_regions_by_edges(pil_image, aspect_ratio_min, aspect_ratio_max, fill_ratio_threshold)
        if len(edge_cards) > len(deduped):
            print(f"[EDGE-DETECT] Found {len(edge_cards)} cards using edges (better than HSV's {len(deduped)})", file=sys.stderr, flush=True)
            return edge_cards
    
    return [(center, warped, bbox, angle) for center, warped, bbox, _, angle in deduped]


def find_card_regions_by_edges(pil_image: Image.Image,
                                aspect_ratio_min: float = 0.50,
                                aspect_ratio_max: float = 0.82,
                                fill_ratio_threshold: float = 0.45) -> List[Tuple[Tuple[int, int], Image.Image, Tuple[int, int, int, int], float]]:
    """Find card-shaped regions using edge detection instead of HSV brightness. Returns (center, warped_image, bbox, angle)."""
    import sys
    rgb = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Canny edge detection with adaptive thresholds
    edges = cv2.Canny(filtered, 30, 150)
    
    # Dilate edges to close gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Save edge detection result for debugging
    import os
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_crops")
    os.makedirs(debug_dir, exist_ok=True)
    Image.fromarray(edges).save(f"{debug_dir}/02_edge_detection.png")
    print(f"[EDGE-DETECT] Saved edge detection to debug_crops/02_edge_detection.png", file=sys.stderr, flush=True)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[EDGE-DETECT] Found {len(contours)} contours", file=sys.stderr, flush=True)
    
    candidates = []
    rejected_by_area = 0
    rejected_by_ratio = 0
    
    # Calculate area thresholds based on image size
    image_area = rgb.shape[0] * rgb.shape[1]
    min_area = max(3500, image_area * 0.002)  # Stricter: was 1500, now 3500 min
    max_area = image_area * 0.08
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            rejected_by_area += 1
            continue
        
        # Get rotated rectangle
        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (width, height), angle_deg = rect
        if width < 30 or height < 30 or width > 5000 or height > 5000:
            continue
        
        # Check aspect ratio
        short_side = min(width, height)
        long_side = max(width, height)
        if long_side == 0:
            continue
        ratio = short_side / long_side
        fill_ratio = area / max(width * height, 1)

        is_bottom_zone = center_y > rgb.shape[0] * 0.58
        ratio_min = aspect_ratio_min
        ratio_max = aspect_ratio_max
        min_fill = fill_ratio_threshold
        if is_bottom_zone:
            ratio_min = max(0.35, aspect_ratio_min - 0.18)
            ratio_max = min(0.95, aspect_ratio_max + 0.18)
            min_fill = max(0.20, fill_ratio_threshold * 0.55)
        
        if not (ratio_min <= ratio <= ratio_max and fill_ratio >= min_fill):
            rejected_by_ratio += 1
            continue
        
        # Rectify card region
        warped = rectify_card_region(rgb, rect)
        x, y, box_w, box_h = cv2.boundingRect(contour)
        bbox = (x, y, x + box_w, y + box_h)
        
        # Normalize angle: cv2.minAreaRect gives -90 to 0 range, convert to our coordinate system
        normalized_angle = -angle_deg if width > height else -(90 + angle_deg)
        
        candidates.append(((center_x, center_y), warped, bbox, area, normalized_angle))
        print(f"[EDGE-ACCEPT] Card at ({int(center_x)},{int(center_y)}): area={area:.0f}, ratio={ratio:.2f}, angle={angle_deg:.1f}° (normalized: {normalized_angle:.1f}°)", file=sys.stderr, flush=True)
    
    # Sort by area, keep largest, deduplicate by proximity (stricter: was 70, now 100 pixels)
    candidates.sort(key=lambda item: item[3], reverse=True)
    deduped = []
    for center, warped, bbox, area, angle in candidates:
        if any(abs(center[0] - prev_center[0]) < 100 and abs(center[1] - prev_center[1]) < 100 for prev_center, _, _, _, _ in deduped):
            continue
        deduped.append((center, warped, bbox, area, angle))
    
    deduped.sort(key=lambda item: (item[0][1], item[0][0]))
    
    print(f"[EDGE-SUMMARY] Rejected: area={rejected_by_area}, ratio={rejected_by_ratio} | Accepted: {len(deduped)}", file=sys.stderr, flush=True)
    
    return [(center, warped, bbox, angle) for center, warped, bbox, _, angle in deduped]


def rotate_for_ocr(pil_image: Image.Image, angle: int) -> Image.Image:
    """Rotate image to help OCR recover tilted ranks and suits."""
    if angle == 0:
        return pil_image
    return pil_image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))


def recognize_card_from_region(card_image: Image.Image, orientations: list = None, 
                                 rank_threshold: float = 0.35, suit_threshold: float = 0.25,
                                 red_ratio_threshold: float = 0.38, manual_rotation: int = 0,
                                 ocr_engine: str = "pytesseract") -> Tuple[Optional[str], float, str]:
    """Recognize a card from a rectified crop using template matching with OCR fallback."""
    if orientations is None:
        orientations = [0, 180]
    
    # Apply manual rotation correction if specified
    if manual_rotation != 0:
        card_image = card_image.rotate(-manual_rotation, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    
    best_card = None
    best_score = 0.0
    best_debug = ""

    for orientation in orientations:
        oriented = rotate_for_ocr(card_image, orientation)
        if oriented.width > oriented.height:
            oriented = oriented.rotate(90, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        oriented = oriented.resize(CARD_CANVAS, Image.LANCZOS)

        # Extract rank and suit from top-left corner with maximum capture area
        # Cards are 180x252 after resize - take generous 40% width x 40% height for both
        rank_crop = oriented.crop((0, 0, int(oriented.width * 0.40), int(oriented.height * 0.30)))
        suit_crop = oriented.crop((0, int(oriented.height * 0.15), int(oriented.width * 0.40), int(oriented.height * 0.50)))
        
        # DEBUG: Save crops for inspection
        import os
        import sys
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_crops")
        os.makedirs(debug_dir, exist_ok=True)
        oriented.save(f"{debug_dir}/card_full_{orientation}.png")
        rank_crop.save(f"{debug_dir}/rank_{orientation}.png")
        suit_crop.save(f"{debug_dir}/suit_{orientation}.png")
        print(f"  Saved debug crops to {debug_dir}/ (card_full, rank, suit @ {orientation}°)", file=sys.stderr, flush=True)

        rank_match, rank_score = template_match_symbol(rank_crop, get_rank_templates(), debug_label=f"RANK@{orientation}°")
        rank_ocr = normalize_rank_symbol(ocr_single_symbol_with_engine(rank_crop, "0123456789TJQKAIO", ocr_engine))
        
        # DEBUG decision logic
        import sys
        print(f"  [DECISION] rank_match={rank_match!r} rank_score={rank_score} rank_ocr={rank_ocr!r}", file=sys.stderr, flush=True)
        print(f"  [DECISION] score<=0.40? {rank_score <= 0.40}, both_exist? {bool(rank_ocr and rank_match)}", file=sys.stderr, flush=True)
        
        # Choose best rank: require very high template confidence to override OCR
        if rank_match and rank_ocr and rank_match == rank_ocr:
            rank = rank_match  # Both agree - definitely correct
        elif rank_match and rank_score >= rank_threshold:
            rank = rank_match  # Template is confident enough, trust it
        elif rank_ocr:
            rank = rank_ocr  # OCR has a result - trust it (template not confident enough or disagrees)
        elif rank_match:
            rank = rank_match  # No OCR, use template
        else:
            rank = rank_ocr  # Last resort OCR

        allowed_suits = RED_SUITS if estimate_red_suit(suit_crop, red_ratio_threshold) else BLACK_SUITS
        suit_match, suit_score = template_match_symbol(suit_crop, get_suit_templates(), allowed_suits, debug_label=f"SUIT@{orientation}° allowed={allowed_suits}")
        suit_ocr = normalize_suit_symbol(ocr_single_symbol_with_engine(suit_crop, "CDHScdhs♣♦♥♠", ocr_engine))
        
        # Choose best suit: prefer template match over OCR generally
        if suit_match and suit_ocr and suit_match == suit_ocr and suit_match in allowed_suits:
            suit = suit_match  # Both agree and in allowed set
        elif suit_match and suit_score >= suit_threshold and suit_match in allowed_suits:
            suit = suit_match  # Template is confident
        elif suit_match and suit_match in allowed_suits and suit_score >= (suit_threshold * 0.4):
            suit = suit_match  # Template has some confidence in allowed suit
        elif suit_ocr and suit_ocr in allowed_suits:
            suit = suit_ocr  # OCR result in allowed set
        elif suit_match:
            suit = suit_match  # Template result even if not in allowed (color detection may be wrong)
        else:
            suit = suit_ocr  # Last resort
        
        # Debug output
        import sys
        print(f"  orient={orientation}° | rank: tpl={rank_match}({rank_score:.3f}) ocr={rank_ocr} → {rank} | suit: tpl={suit_match}({suit_score:.3f}) ocr={suit_ocr} allowed={allowed_suits} → {suit}", file=sys.stderr, flush=True)

        if rank and suit:
            # Weight rank more than suit and mildly penalize 180° orientation.
            # This avoids upside-down matches winning due an oversized suit symbol score.
            orientation_penalty = 0.20 if orientation == 180 else 0.0
            score = (rank_score * 1.6) + (suit_score * 0.7) - orientation_penalty
            if score > best_score:
                best_card = rank + suit
                best_score = score
                best_debug = (
                    f"orientation={orientation} rank={rank}({rank_score:.2f}/{rank_ocr or '-'}) "
                    f"suit={suit}({suit_score:.2f}/{suit_ocr or '-'})"
                )
                # Early exit if we have high confidence
                if score > 1.55:
                    break

    return best_card, best_score, best_debug


def detect_cards_by_regions(pil_image: Image.Image, orientations: list = None,
                             rank_threshold: float = 0.60, suit_threshold: float = 0.20,
                             white_ratio_threshold: float = 0.24,
                             aspect_ratio_min: float = 0.45, aspect_ratio_max: float = 0.88,
                             fill_ratio_threshold: float = 0.38, red_ratio_threshold: float = 0.38,
                             hole_rotation: int = 0, board_rotation: int = 0,
                             card1_rotation: int = 0, card2_rotation: int = 0,
                             hsv_v_min: int = 95, hsv_v_max: int = 255,
                             hsv_s_max: int = 75,
                             ocr_engine: str = "pytesseract") -> Tuple[List[str], List[str], List[int]]:
    """Detect cards from likely card regions using template matching and region-local OCR. Returns (cards, debug_lines, angles)."""
    if orientations is None:
        orientations = [0, 180]
    cards = []
    angles = []
    debug_lines = []
    detections = []

    for index, (center, card_image, bbox, detected_angle) in enumerate(
        find_card_regions(pil_image, white_ratio_threshold, aspect_ratio_min, aspect_ratio_max, fill_ratio_threshold, hsv_v_min, hsv_v_max, hsv_s_max), 
        start=1):
        # Use detected angle from edge detection, or fall back to manual rotation if set
        if index == 1 and card1_rotation != 0:
            manual_rotation = card1_rotation
        elif index == 2 and card2_rotation != 0:
            manual_rotation = card2_rotation
        elif index > 2 and board_rotation != 0:
            manual_rotation = board_rotation
        else:
            # Use auto-detected angle from edge detection
            manual_rotation = int(detected_angle)
            if manual_rotation != 0:
                print(f"[AUTO-ANGLE] Card {index}: using auto-detected angle {manual_rotation}°", file=sys.stderr, flush=True)

        box_w, box_h = bbox[2], bbox[3]
        is_bottom_zone = center[1] > (pil_image.height * 0.42)
        
        # Ignore tiny bottom-zone regions (usually isolated suit/rank symbols, not cards).
        if is_bottom_zone and box_h < int(pil_image.height * 0.16):
            debug_lines.append(f"region {index} @ {center}: skipped tiny bottom contour (w={box_w}, h={box_h})")
            continue

        # If a bottom-zone region is unusually wide, it is often two overlapped/fanned hole cards.
        # Split it into left/right candidates and run recognition for each side.
        is_wide_bottom_blob = (
            is_bottom_zone
            and box_h >= int(pil_image.height * 0.20)
            and box_w >= int(box_h * 0.95)
        )
        if is_wide_bottom_blob:
            w, h = card_image.size
            overlap = int(w * 0.10)
            mid = w // 2
            left_crop = card_image.crop((0, 0, min(w, mid + overlap), h))
            right_crop = card_image.crop((max(0, mid - overlap), 0, w, h))

            # Blob angle can be unstable for fanned hole cards. If it's too steep,
            # do not apply it to split-side recognition.
            split_rotation = manual_rotation if abs(manual_rotation) <= 25 else 0

            split_debug = []
            for side_name, side_img, side_shift in (("L", left_crop, -18), ("R", right_crop, 18)):
                side_card, side_score, side_info = recognize_card_from_region(
                    side_img, orientations, rank_threshold, suit_threshold, red_ratio_threshold, split_rotation, ocr_engine
                )
                # Split results are noisy; require stronger confidence than normal regions.
                if side_card and side_score >= 1.20:
                    side_center = (center[0] + side_shift, center[1])
                    detections.append((side_center, side_card, side_score, bbox, f"[HOLE-SPLIT-{side_name}] {side_info}", split_rotation))
                    split_debug.append(f"{side_name}:{side_card}@{side_score:.2f}")

            if split_debug:
                debug_lines.append(f"region {index} @ {center}: split hole-card blob -> {' | '.join(split_debug)}")
                continue

        card, score, debug = recognize_card_from_region(card_image, orientations, rank_threshold, suit_threshold, red_ratio_threshold, manual_rotation, ocr_engine)
        if not card:
            debug_lines.append(f"region {index} @ {center}: no card match")
            continue
        detections.append((center, card, score, bbox, debug, manual_rotation))

    detections.sort(key=lambda item: item[2], reverse=True)
    if len(detections) > 7:
        detections = detections[:7]
    detections.sort(key=lambda item: (item[0][1], item[0][0]))

    for center, card, score, bbox, debug, angle in detections:
        if card not in cards:
            cards.append(card)
            angles.append(angle)
        debug_lines.append(f"region {center} bbox={bbox} -> {card} score={score:.2f} angle={angle}° {debug}")

    return cards, debug_lines, angles


def ocr_cards_from_image(pil_image: Image.Image, rotation_angles: list = None, 
                          orientations: list = None, upscale_factor: int = 2,
                          rank_threshold: float = 0.60, suit_threshold: float = 0.20,
                          white_ratio_threshold: float = 0.24,
                          aspect_ratio_min: float = 0.45, aspect_ratio_max: float = 0.88,
                          fill_ratio_threshold: float = 0.38, red_ratio_threshold: float = 0.38,
                          hole_rotation: int = 0, board_rotation: int = 0,
                          card1_rotation: int = 0, card2_rotation: int = 0,
                          hsv_v_min: int = 95, hsv_v_max: int = 255,
                         hsv_s_max: int = 75,
                         ocr_engine: str = "pytesseract") -> Tuple[str, List[str], List[int]]:
    """Use region detection first, then multi-angle OCR as a fallback and supplement. Returns (raw_text, cards, angles)."""
    if rotation_angles is None:
        rotation_angles = [-5, 0, 5]
    if orientations is None:
        orientations = [0, 180]
    
    region_cards, region_debug, region_angles = detect_cards_by_regions(pil_image, orientations, rank_threshold, suit_threshold, 
                                                          white_ratio_threshold, aspect_ratio_min, aspect_ratio_max,
                                                          fill_ratio_threshold, red_ratio_threshold,
                                                          hole_rotation, board_rotation, card1_rotation, card2_rotation,
                                                          hsv_v_min, hsv_v_max, hsv_s_max, ocr_engine)
    all_cards = list(region_cards)
    all_angles = list(region_angles)
    
    # Try full-image OCR unless region detection already found a near-complete set.
    # This handles cases where cards have glows/highlights that break region detection
    if len(region_cards) >= 5:
        preview_lines = [f"Region detector found: {', '.join(region_cards)}"]
        preview_lines.extend(region_debug[:8])
        preview_lines.append("Skipped full-image OCR (sufficient cards from regions)")
        return "\n".join(preview_lines), region_cards, region_angles
    
    best_cards: List[str] = []
    best_text = ""
    best_angle = 0
    best_score = (-1, 0)

    for angle in rotation_angles:
        rotated = rotate_for_ocr(pil_image, angle)
        processed = preprocess_for_ocr(rotated, upscale_factor)
        text = ocr_text_with_engine(processed, engine=ocr_engine, psm=11)
        # DEBUG: Show what OCR actually read
        import sys
        print(f"  [FULL-OCR@{angle}°] Raw text: {text.strip()!r}", file=sys.stderr, flush=True)
        cards = parse_cards_from_text(text)

        for card in cards:
            if card not in all_cards and len(all_cards) < 7:
                all_cards.append(card)

        score = (len(cards), -abs(angle))
        if score > best_score:
            best_score = score
            best_text = text
            best_cards = cards
            best_angle = angle

    merged_cards = all_cards if all_cards else best_cards
    preview_lines = [f"Region detector found: {', '.join(region_cards) if region_cards else 'none'}"]
    preview_lines.extend(region_debug[:8])
    preview_lines.append(f"Best full-image OCR angle: {best_angle}°")
    if merged_cards and merged_cards != best_cards:
        preview_lines.append(f"Merged cards: {', '.join(merged_cards)}")

    # Pad angles list to match cards count
    if all_angles:
        while len(all_angles) < len(merged_cards):
            all_angles.append(0)  # Default angle 0 for cards without detected angle
        final_angles = all_angles[:len(merged_cards)]
    else:
        final_angles = [0] * len(merged_cards)
    
    return "\n".join(preview_lines) + "\n" + best_text, merged_cards, final_angles


def parse_cards_from_text(text: str) -> List[str]:
    """
    Extract card tokens like 'Ah', 'Kd', '2s', 'Tc' from OCR output.
    Returns a list of canonical card strings e.g. ['Ah', 'Kd'].
    """
    text = text.upper().replace("♣", "C").replace("♦", "D").replace("♥", "H").replace("♠", "S")

    # Match patterns like: A H, K D, 10H, TH, 1OH, IOH, 2 S ...
    pattern = re.compile(r'\b(10|1O|IO|[2-9TJQKA])\s*([CDHS])\b', re.IGNORECASE)
    cards = []
    for m in pattern.finditer(text):
        rank_raw = m.group(1).upper()
        suit_raw = m.group(2).lower()
        rank = RANK_ALIASES.get(rank_raw, rank_raw)
        suit = SUIT_ALIASES.get(suit_raw, suit_raw)
        if rank in RANKS and suit in SUITS:
            card = rank + suit
            if card not in cards:
                cards.append(card)
    return cards


def capture_and_ocr(rotation_angles: list = None, orientations: list = None, 
                    upscale_factor: int = 2, rank_threshold: float = 0.60, 
                    suit_threshold: float = 0.20, white_ratio_threshold: float = 0.24,
                    aspect_ratio_min: float = 0.45, aspect_ratio_max: float = 0.88,
                    fill_ratio_threshold: float = 0.38, red_ratio_threshold: float = 0.38,
                    hole_rotation: int = 0, board_rotation: int = 0,
                    card1_rotation: int = 0, card2_rotation: int = 0,
                    hsv_v_min: int = 95, hsv_v_max: int = 255,
                    hsv_s_max: int = 75,
                    ocr_engine: str = "pytesseract",
                    auto_expand_bbox: bool = False,
                    bbox: tuple = None) -> Tuple[str, List[str], List[int]]:
    """Capture screen region (or full screen), OCR multiple tilt angles, and return the best text plus merged cards."""
    import sys
    print(f"[CAPTURE] bbox={bbox}", file=sys.stderr, flush=True)
    
    # Always grab full screen, then crop manually (more reliable on macOS)
    screenshot = ImageGrab.grab()
    print(f"[CAPTURE] full screenshot size={screenshot.size}", file=sys.stderr, flush=True)
    
    if bbox:
        x1, y1, x2, y2 = bbox
        sx, sy = screenshot.size

        if auto_expand_bbox:
            # Optional expansion for partially visible/fanned hole cards.
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            expand_x = int(w * 0.12)
            expand_up = int(h * 0.08)
            expand_down = int(h * 0.45)
            x1 = max(0, x1 - expand_x)
            x2 = min(sx, x2 + expand_x)
            y1 = max(0, y1 - expand_up)
            y2 = min(sy, y2 + expand_down)
            adjusted_bbox = (x1, y1, x2, y2)
            print(f"[CAPTURE] adjusted bbox={adjusted_bbox}", file=sys.stderr, flush=True)
        else:
            adjusted_bbox = (
                max(0, min(sx, int(x1))),
                max(0, min(sy, int(y1))),
                max(0, min(sx, int(x2))),
                max(0, min(sy, int(y2))),
            )
            print(f"[CAPTURE] strict bbox={adjusted_bbox}", file=sys.stderr, flush=True)

        # Crop to adjusted/strict region (bbox is in native/Retina coordinates)
        screenshot = screenshot.crop(adjusted_bbox)
        print(f"[CAPTURE] cropped to bbox, new size={screenshot.size}", file=sys.stderr, flush=True)
    
    print(f"[CAPTURE] final screenshot size={screenshot.size}", file=sys.stderr, flush=True)
    return ocr_cards_from_image(screenshot, rotation_angles, orientations, upscale_factor, 
                                 rank_threshold, suit_threshold, white_ratio_threshold,
                                 aspect_ratio_min, aspect_ratio_max, fill_ratio_threshold, red_ratio_threshold,
                                 hole_rotation, board_rotation, card1_rotation, card2_rotation,
                                 hsv_v_min, hsv_v_max, hsv_s_max, ocr_engine)

# ──────────────────────────────────────────────
# Poker hand evaluator (pure Python, no external lib)
# ──────────────────────────────────────────────

def card_rank(card: str) -> int:
    return RANK_VALUES[card[0]]


def card_suit(card: str) -> str:
    return card[1]


def hand_rank(hand: List[str]) -> tuple:
    """
    Evaluate a 5-card hand.
    Returns a tuple (rank_class, tiebreakers...) where higher is better.
    rank_class: 8=straight flush, 7=four of a kind, 6=full house,
                5=flush, 4=straight, 3=three of a kind, 2=two pair,
                1=one pair, 0=high card
    """
    ranks = sorted([card_rank(c) for c in hand], reverse=True)
    suits = [card_suit(c) for c in hand]

    is_flush = len(set(suits)) == 1
    is_straight = (len(set(ranks)) == 5 and ranks[0] - ranks[4] == 4)
    # Ace-low straight: A-2-3-4-5 → ranks would be [12,3,2,1,0]
    if set(ranks) == {12, 3, 2, 1, 0}:
        is_straight = True
        ranks = [3, 2, 1, 0, -1]  # treat Ace as low

    from collections import Counter
    counts = Counter(ranks)
    groups = sorted(counts.values(), reverse=True)
    rank_groups = sorted(counts.keys(), key=lambda r: (counts[r], r), reverse=True)

    if is_straight and is_flush:
        return (8, ranks[0])
    if groups == [4, 1]:
        return (7, *rank_groups)
    if groups == [3, 2]:
        return (6, *rank_groups)
    if is_flush:
        return (5, *ranks)
    if is_straight:
        return (4, ranks[0])
    if groups[0] == 3:
        return (3, *rank_groups)
    if groups[:2] == [2, 2]:
        return (2, *rank_groups)
    if groups[0] == 2:
        return (1, *rank_groups)
    return (0, *ranks)


def best_5_of_7(cards: List[str]) -> tuple:
    """Return the best hand_rank from all 5-card combos of up to 7 cards."""
    if len(cards) < 5:
        return (-1,)
    best = max(hand_rank(list(combo)) for combo in combinations(cards, 5))
    return best


HAND_NAMES = {
    8: "Straight Flush", 7: "Four of a Kind", 6: "Full House",
    5: "Flush", 4: "Straight", 3: "Three of a Kind",
    2: "Two Pair", 1: "One Pair", 0: "High Card",
}


# ──────────────────────────────────────────────
# Monte-Carlo equity estimator
# ──────────────────────────────────────────────

import random


def monte_carlo_equity(hole: List[str], board: List[str],
                       num_opponents: int = 1, iterations: int = 2000) -> float:
    """
    Estimate win equity via Monte Carlo simulation.
    Returns probability [0,1] of winning (ties count as 0.5).
    """
    if len(hole) < 2:
        return 0.0

    deck = [r + s for r in RANKS for s in SUITS
            if (r + s) not in hole and (r + s) not in board]

    wins = 0.0
    for _ in range(iterations):
        random.shuffle(deck)
        idx = 0
        # Deal community cards to complete 5 on board
        community = list(board)
        while len(community) < 5:
            community.append(deck[idx]); idx += 1

        # Deal opponent hole cards
        opponents = []
        for _ in range(num_opponents):
            opp = [deck[idx], deck[idx + 1]]; idx += 2
            opponents.append(opp)

        my_best = best_5_of_7(hole + community)

        beat_all = True
        for opp in opponents:
            opp_best = best_5_of_7(opp + community)
            if opp_best > my_best:
                beat_all = False
                break
            if opp_best == my_best:
                # Tie – count half
                beat_all = None
                break

        if beat_all is True:
            wins += 1.0
        elif beat_all is None:
            wins += 0.5

    return wins / iterations


# ──────────────────────────────────────────────
# GTO strategy advisor
# ──────────────────────────────────────────────

POSITION_ORDER = ["UTG", "UTG+1", "MP", "HJ", "CO", "BTN", "SB", "BB"]


def gto_advice(hole: List[str], board: List[str],
               equity: float, street: str,
               position: str, pot_odds: Optional[float]) -> dict:
    """
    Return a GTO-aligned strategy recommendation.
    This is a simplified rule-based model informed by GTO principles.
    """
    advice = {}
    hand_str = ""
    my_best = best_5_of_7(hole + board)
    hand_class = my_best[0] if my_best != (-1,) else -1

    if hand_class >= 0:
        hand_str = HAND_NAMES.get(hand_class, "Unknown")

    advice["hand_made"] = hand_str if hand_str else "Incomplete (need more cards)"

    # Pot odds decision threshold
    if pot_odds is not None and pot_odds > 0:
        required_equity = pot_odds / (1 + pot_odds)
        advice["pot_odds_threshold"] = f"{required_equity:.1%}"
        pot_odds_ok = equity >= required_equity
    else:
        pot_odds_ok = None

    # Pre-flop hand strength heuristic
    if street == "Pre-Flop":
        r1, r2 = card_rank(hole[0]), card_rank(hole[1])
        s1, s2 = card_suit(hole[0]), card_suit(hole[1])
        suited = s1 == s2
        gap = abs(r1 - r2)
        high = max(r1, r2)
        pair = r1 == r2

        if pair and high >= 10:  # TT+
            strength = "premium"
        elif pair and high >= 6:  # 66-99
            strength = "strong"
        elif pair:
            strength = "speculative"
        elif high >= 12 and (r1 + r2) >= 23:  # AK, AQ, KQ
            strength = "premium"
        elif high >= 11 and gap <= 1:
            strength = "strong"
        elif suited and gap <= 2 and high >= 9:
            strength = "playable"
        elif (r1 + r2) >= 18 and gap <= 3:
            strength = "playable"
        else:
            strength = "weak"

        late_position = position in ("BTN", "CO", "HJ") if position else False
        early_position = position in ("UTG", "UTG+1") if position else False

        if strength == "premium":
            advice["action"] = "Raise / 3-Bet aggressively"
            advice["reasoning"] = "Premium holding — build the pot and narrow the field."
        elif strength == "strong":
            if late_position:
                advice["action"] = "Raise / Call 3-Bet"
                advice["reasoning"] = "Strong hand with position advantage, open or call re-raises."
            else:
                advice["action"] = "Raise / Fold to 4-Bet"
                advice["reasoning"] = "Solid hand but proceed cautiously from early position."
        elif strength == "playable":
            if late_position:
                advice["action"] = "Open Raise or Call"
                advice["reasoning"] = "Speculative hand best played in position."
            else:
                advice["action"] = "Fold or Limp"
                advice["reasoning"] = "Marginal from early position; avoid bloated pots."
        else:
            advice["action"] = "Fold"
            advice["reasoning"] = "Weak holding — not worth investing chips."

    else:
        # Post-flop: equity-based recommendation
        if equity >= 0.70:
            advice["action"] = "Bet / Raise (Value)"
            advice["reasoning"] = f"Strong equity ({equity:.1%}). Extract maximum value."
        elif equity >= 0.50:
            advice["action"] = "Bet or Check-Call"
            advice["reasoning"] = f"Ahead on average ({equity:.1%}). Bet for value or call bets."
        elif equity >= 0.35:
            if pot_odds_ok is True:
                advice["action"] = "Call (pot odds justify)"
                advice["reasoning"] = (
                    f"Equity {equity:.1%} marginally ahead of required "
                    f"{advice.get('pot_odds_threshold', '?')}."
                )
            elif pot_odds_ok is False:
                advice["action"] = "Fold (poor pot odds)"
                advice["reasoning"] = (
                    f"Equity {equity:.1%} below required pot odds threshold "
                    f"{advice.get('pot_odds_threshold', '?')}."
                )
            else:
                advice["action"] = "Check or Small Bet (Draw)"
                advice["reasoning"] = f"Drawing equity {equity:.1%}. Keep pot small or semi-bluff."
        elif equity >= 0.20:
            if pot_odds_ok is True:
                advice["action"] = "Call (pot odds)"
                advice["reasoning"] = (
                    f"Equity {equity:.1%} meets pot odds requirement "
                    f"{advice.get('pot_odds_threshold', '?')}."
                )
            else:
                advice["action"] = "Fold"
                advice["reasoning"] = f"Equity {equity:.1%} too low without correct pot odds."
        else:
            advice["action"] = "Fold or Bluff (polarised range)"
            advice["reasoning"] = (
                f"Very low equity ({equity:.1%}). Only continue as a bluff "
                "if representing a strong range."
            )

    advice["equity"] = equity
    return advice


# ──────────────────────────────────────────────
# Tkinter GUI  (pure tk – no ttk, works on all platforms)
# ──────────────────────────────────────────────

class PokerGeniusApp(tk.Tk):

    BG  = "#1a1a2e"
    BG2 = "#16213e"
    FG  = "#e0e0e0"
    ACC = "#f0c040"
    GRN = "#40ff80"
    DIM = "#888888"

    # widget factory helpers ─────────────────────
    def _lf(self, parent, text):
        return tk.LabelFrame(parent, text=text,
                             bg=self.BG, fg=self.ACC,
                             font=("Helvetica", 11, "bold"),
                             bd=1, relief=tk.GROOVE, padx=8, pady=6)

    def _label(self, parent, text, **kw):
        font = kw.pop('font', ("Helvetica", 11))
        fg = kw.pop('fg', self.FG)
        bg = kw.pop('bg', self.BG)
        return tk.Label(parent, text=text,
                        bg=bg, fg=fg,
                        font=font, **kw)

    def _btn(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Helvetica", 11, "bold"),
                         relief=tk.RAISED, bd=2,
                         padx=10, pady=4)

    def _entry(self, parent, textvariable, width):
        return tk.Entry(parent, textvariable=textvariable, width=width,
                        bg=self.BG2, fg=self.FG, insertbackground=self.FG,
                        relief=tk.SUNKEN, bd=1, font=("Helvetica", 11))

    def _text(self, parent, height):
        return scrolledtext.ScrolledText(
            parent, height=height,
            bg=self.BG2, fg=self.FG, insertbackground=self.FG,
            font=("Courier", 11), relief=tk.FLAT, wrap=tk.WORD)

    # ────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.title("Poker Genius – GTO Advisor")
        self.geometry("1200x850")
        self.minsize(900, 700)
        self.configure(bg=self.BG)
        
        # OCR Parameters (configurable)
        self.ocr_rotation_angles = [-5, 0, 5]
        self.ocr_angle_step = 5
        self.ocr_orientations = [0, 180]
        self.ocr_engine = "pytesseract"
        self.auto_expand_bbox = False  # Respect selected board area strictly by default
        self.ocr_rank_threshold = 0.60  # Balanced OCR/template blending
        self.ocr_suit_threshold = 0.20  # Slightly stricter than rank for symbol stability
        self.ocr_upscale_factor = 2
        
        # Card region detection thresholds
        self.white_ratio_threshold = 0.24  # Balanced for white board cards + partially visible hole cards
        self.aspect_ratio_min = 0.45  # Minimum aspect ratio (short/long side) for card shape
        self.aspect_ratio_max = 0.88  # Maximum aspect ratio (short/long side) for card shape
        self.fill_ratio_threshold = 0.38  # Minimum fill ratio (area/bounding_box) for card shape
        self.red_ratio_threshold = 0.38  # Red channel ratio to classify suit as red
        self.hsv_v_min = 95  # HSV brightness (V channel) minimum threshold for card detection
        self.hsv_v_max = 255  # HSV brightness (V channel) maximum threshold for card detection
        self.hsv_s_max = 75  # HSV saturation (S channel) maximum threshold for white-ish card surfaces
        
        # Manual rotation angles for card correction
        self.hole_card_rotation = 0  # Rotation angle for hole cards (degrees)
        self.board_card_rotation = 0  # Rotation angle for board cards (degrees)
        
        # Debug image previews for OCR params window
        self.debug_card_image = None  # Most recent card crop
        self.debug_rank_image = None  # Most recent rank crop
        self.debug_suit_image = None  # Most recent suit crop
        self.debug_hsv_mask = None   # Most recent HSV mask
        self.preview_refresh_callback = None  # Callback to refresh preview window
        
        # Screen capture region (None = full screen)
        self.capture_bbox = None
        
        self._build_ui()

    def _build_ui(self):
        B, F = self.BG, self.FG

        pad = dict(padx=12, pady=6)

        # ── Title ──────────────────────────────────
        tk.Label(self, text="  Poker Genius – GTO Advisor  ",
                 bg=B, fg=self.ACC,
                 font=("Helvetica", 16, "bold")).pack(pady=(14, 4))

        # ── Controls row ───────────────────────────
        ctrl = tk.Frame(self, bg=B)
        ctrl.pack(fill=tk.X, **pad)

        self._label(ctrl, "Position:").pack(side=tk.LEFT)
        self.position_var = tk.StringVar(value="BTN")
        pos_menu = tk.OptionMenu(ctrl, self.position_var, *POSITION_ORDER)
        pos_menu.config(bg=self.BG2, fg=F, activebackground=B,
                        font=("Helvetica", 11), relief=tk.FLAT, highlightthickness=0)
        pos_menu["menu"].config(bg=self.BG2, fg=F, font=("Helvetica", 11))
        pos_menu.pack(side=tk.LEFT, padx=(2, 16))

        self._label(ctrl, "Opponents:").pack(side=tk.LEFT)
        self.opponents_var = tk.IntVar(value=1)
        tk.Spinbox(ctrl, from_=1, to=8, width=3,
                   textvariable=self.opponents_var,
                   bg=self.BG2, fg=F, insertbackground=F,
                   buttonbackground=self.BG2,
                   font=("Helvetica", 11), relief=tk.SUNKEN
                   ).pack(side=tk.LEFT, padx=(2, 16))

        self._label(ctrl, "Pot odds (call/pot):").pack(side=tk.LEFT)
        self.pot_odds_var = tk.StringVar(value="")
        self._entry(ctrl, self.pot_odds_var, 7).pack(side=tk.LEFT, padx=(2, 0))

        # ── Street selector ────────────────────────
        sf = tk.Frame(self, bg=B)
        sf.pack(fill=tk.X, padx=12, pady=2)
        self._label(sf, "Street:").pack(side=tk.LEFT)
        self.street_var = tk.StringVar(value="Auto-detect")
        for s in ("Auto-detect", "Pre-Flop", "Flop", "Turn", "River"):
            tk.Radiobutton(sf, text=s, variable=self.street_var, value=s,
                           bg=B, fg=F, selectcolor=self.BG2,
                           activebackground=B, activeforeground=self.ACC,
                           font=("Helvetica", 11)).pack(side=tk.LEFT, padx=3)

        # ── Manual card input ──────────────────────
        mf = self._lf(self, "Manual card input  (e.g.  Ah Kd  /  7s 8h Qd)")
        mf.pack(fill=tk.X, padx=12, pady=6)

        inner = tk.Frame(mf, bg=B)
        inner.pack(fill=tk.X)
        self._label(inner, "Hole cards:").grid(row=0, column=0, sticky=tk.W, padx=4, pady=2)
        self.hole_entry = self._entry(inner, tk.StringVar(), 14)
        self.hole_entry.grid(row=0, column=1, sticky=tk.W, padx=4)
        self._label(inner, "Board cards:").grid(row=0, column=2, sticky=tk.W, padx=(16, 4))
        self.board_entry = self._entry(inner, tk.StringVar(), 20)
        self.board_entry.grid(row=0, column=3, sticky=tk.W, padx=4)
        
        # Manual rotation controls
        # Initialize rotation variables (controlled via Set Card Angles overlay)
        self.hole_rotation_var = tk.IntVar(value=0)  # Rotation for hole cards (used for both)
        self.board_rotation_var = tk.IntVar(value=0)  # Rotation for board cards
        self.card1_rotation = 0  # Individual rotation for first hole card
        self.card2_rotation = 0  # Individual rotation for second hole card

        # ── Buttons ────────────────────────────────
        bf = tk.Frame(self, bg=B)
        bf.pack(pady=6)
        self.capture_btn = self._btn(bf, "Capture Screen & Analyze", self._on_capture)
        self.capture_btn.pack(side=tk.LEFT, padx=6)
        self._btn(bf, "Analyze Manual Input", self._on_manual).pack(side=tk.LEFT, padx=6)
        self._btn(bf, "Clear", self._on_clear).pack(side=tk.LEFT, padx=6)
        self.board_area_btn = self._btn(bf, "Set Board Area", self._set_board_area)
        self.board_area_btn.pack(side=tk.LEFT, padx=6)
        self.card_angles_btn = self._btn(bf, "Set Card Angles", self._set_card_angles)
        self.card_angles_btn.pack(side=tk.LEFT, padx=6)
        self._btn(bf, "OCR Parameters", self._open_ocr_params).pack(side=tk.LEFT, padx=6)

        # ── OCR output ─────────────────────────────
        ocr_lf = self._lf(self, "OCR / Detected cards")
        ocr_lf.pack(fill=tk.X, padx=12, pady=4)
        self.ocr_text = scrolledtext.ScrolledText(
            ocr_lf, height=3,
            bg=self.BG2, fg="#a0d0ff", insertbackground=F,
            font=("Courier", 10), relief=tk.FLAT, wrap=tk.WORD)
        self.ocr_text.pack(fill=tk.X)

        # ── Strategy output ────────────────────────
        st_lf = self._lf(self, "Strategy Recommendation")
        st_lf.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        self.action_label = tk.Label(st_lf, text="—",
                                     bg=B, fg=self.GRN,
                                     font=("Helvetica", 13, "bold"))
        self.action_label.pack(anchor=tk.W, pady=(0, 4))

        self.strategy_text = scrolledtext.ScrolledText(
            st_lf, height=9,
            bg=self.BG2, fg=F, insertbackground=F,
            font=("Courier", 11), relief=tk.FLAT, wrap=tk.WORD)
        self.strategy_text.pack(fill=tk.BOTH, expand=True)

        # ── Status bar ─────────────────────────────
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(self, textvariable=self.status_var,
                 bg=B, fg=self.DIM,
                 font=("Helvetica", 10), anchor=tk.W).pack(
                     fill=tk.X, padx=14, pady=(2, 8))

    # ── helpers ──────────────────────────────────
    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    def _set_pending_result(self, action: str, details: str):
        self.action_label.config(text=action)
        self.strategy_text.delete("1.0", tk.END)
        self.strategy_text.insert(tk.END, details)

    # ── event handlers ───────────────────────────
    def _on_clear(self):
        self.ocr_text.delete("1.0", tk.END)
        self.strategy_text.delete("1.0", tk.END)
        self.action_label.config(text="—")
        self.hole_entry.delete(0, tk.END)
        self.board_entry.delete(0, tk.END)
        self._set_status("Cleared.")
    
    def _open_ocr_params(self):
        """Open window to configure OCR parameters."""
        win = tk.Toplevel(self)
        win.title("OCR Parameters")
        win.geometry("1000x950")
        win.configure(bg=self.BG)
        win.transient(self)
        
        B, F = self.BG, self.FG
        
        tk.Label(win, text="OCR Recognition Parameters", 
                 bg=B, fg=self.ACC, font=("Helvetica", 14, "bold")).pack(pady=12)
        
        # Create scrollable container
        canvas_frame = tk.Frame(win, bg=B)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        
        canvas = tk.Canvas(canvas_frame, bg=B, highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=B)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Main content frame (now inside scrollable_frame)
        content = tk.Frame(scrollable_frame, bg=B)
        content.pack(fill=tk.X, padx=20, pady=10)
        
        # Rotation angles
        tk.Label(content, text="Rotation Angles (degrees):", 
                 bg=B, fg=F, font=("Helvetica", 11, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0,4))
        tk.Label(content, text="Number of angles to test for tilted cards", 
                 bg=B, fg=self.DIM, font=("Helvetica", 9)).grid(row=1, column=0, sticky=tk.W, pady=(0,8))
        
        angles_frame = tk.Frame(content, bg=B)
        angles_frame.grid(row=2, column=0, sticky=tk.W, pady=(0,16))
        
        tk.Label(angles_frame, text="From:", bg=B, fg=F).pack(side=tk.LEFT, padx=(0,4))
        angle_min_var = tk.IntVar(value=min(self.ocr_rotation_angles))
        tk.Spinbox(angles_frame, from_=-15, to=0, width=4, textvariable=angle_min_var,
                   bg=self.BG2, fg=F, insertbackground=F).pack(side=tk.LEFT, padx=2)
        
        tk.Label(angles_frame, text="To:", bg=B, fg=F).pack(side=tk.LEFT, padx=(12,4))
        angle_max_var = tk.IntVar(value=max(self.ocr_rotation_angles))
        tk.Spinbox(angles_frame, from_=0, to=15, width=4, textvariable=angle_max_var,
                   bg=self.BG2, fg=F, insertbackground=F).pack(side=tk.LEFT, padx=2)
        
        tk.Label(angles_frame, text="Step:", bg=B, fg=F).pack(side=tk.LEFT, padx=(12,4))
        angle_step_var = tk.IntVar(value=self.ocr_angle_step)
        tk.Spinbox(angles_frame, from_=1, to=10, width=3, textvariable=angle_step_var,
                   bg=self.BG2, fg=F, insertbackground=F).pack(side=tk.LEFT, padx=2)
        
        # Card orientations
        tk.Label(content, text="Card Orientations:", 
                 bg=B, fg=F, font=("Helvetica", 11, "bold")).grid(row=3, column=0, sticky=tk.W, pady=(0,4))
        tk.Label(content, text="Which rotations to test per card region", 
                 bg=B, fg=self.DIM, font=("Helvetica", 9)).grid(row=4, column=0, sticky=tk.W, pady=(0,8))
        
        orient_frame = tk.Frame(content, bg=B)
        orient_frame.grid(row=5, column=0, sticky=tk.W, pady=(0,16))
        
        orient_vars = {}
        for angle, label in [(0, "0° (upright)"), (90, "90°"), (180, "180° (inverted)"), (270, "270°")]:
            var = tk.BooleanVar(value=(angle in self.ocr_orientations))
            orient_vars[angle] = var
            tk.Checkbutton(orient_frame, text=label, variable=var, 
                          bg=B, fg=F, selectcolor=self.BG2, 
                          activebackground=B, activeforeground=F).pack(anchor=tk.W)
        
        # Confidence thresholds
        tk.Label(content, text="Template Match Thresholds:", 
                 bg=B, fg=F, font=("Helvetica", 11, "bold")).grid(row=6, column=0, sticky=tk.W, pady=(0,4))
        tk.Label(content, text="Higher = trust OCR more (template needs higher confidence to override)", 
                 bg=B, fg=self.DIM, font=("Helvetica", 9)).grid(row=7, column=0, sticky=tk.W, pady=(0,8))
        
        thresh_frame = tk.Frame(content, bg=B)
        thresh_frame.grid(row=8, column=0, sticky=tk.W, pady=(0,16))
        
        tk.Label(thresh_frame, text="Rank:", bg=B, fg=F).pack(side=tk.LEFT, padx=(0,4))
        rank_thresh_var = tk.DoubleVar(value=self.ocr_rank_threshold)
        tk.Scale(thresh_frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                variable=rank_thresh_var, bg=B, fg=F, highlightthickness=0, 
                troughcolor=self.BG2, length=100).pack(side=tk.LEFT, padx=2)
        
        tk.Label(thresh_frame, text="Suit:", bg=B, fg=F).pack(side=tk.LEFT, padx=(12,4))
        suit_thresh_var = tk.DoubleVar(value=self.ocr_suit_threshold)
        tk.Scale(thresh_frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                variable=suit_thresh_var, bg=B, fg=F, highlightthickness=0,
                troughcolor=self.BG2, length=100).pack(side=tk.LEFT, padx=2)
        
        # Upscale factor
        tk.Label(content, text="Image Preprocessing:", 
                 bg=B, fg=F, font=("Helvetica", 11, "bold")).grid(row=9, column=0, sticky=tk.W, pady=(0,4))
        
        upscale_frame = tk.Frame(content, bg=B)
        upscale_frame.grid(row=10, column=0, sticky=tk.W, pady=(0,8))
        
        tk.Label(upscale_frame, text="Upscale factor:", bg=B, fg=F).pack(side=tk.LEFT, padx=(0,4))
        upscale_var = tk.IntVar(value=self.ocr_upscale_factor)
        tk.Spinbox(upscale_frame, from_=1, to=4, width=3, textvariable=upscale_var,
                   bg=self.BG2, fg=F, insertbackground=F).pack(side=tk.LEFT, padx=2)
        tk.Label(upscale_frame, text="×  (higher = slower but more accurate)", 
                 bg=B, fg=self.DIM, font=("Helvetica", 9)).pack(side=tk.LEFT, padx=4)

        # OCR engine selection
        engine_frame = tk.Frame(content, bg=B)
        engine_frame.grid(row=11, column=0, sticky=tk.W, pady=(0,12))
        tk.Label(engine_frame, text="OCR engine:", bg=B, fg=F).pack(side=tk.LEFT, padx=(0,6))
        ocr_engine_var = tk.StringVar(value=self.ocr_engine)
        engine_menu = tk.OptionMenu(engine_frame, ocr_engine_var, *OCR_ENGINES)
        engine_menu.config(bg=self.BG2, fg=F, activebackground=B,
                  font=("Helvetica", 10), relief=tk.FLAT, highlightthickness=0)
        engine_menu["menu"].config(bg=self.BG2, fg=F, font=("Helvetica", 10))
        engine_menu.pack(side=tk.LEFT)
        tk.Label(engine_frame, text="(easyocr may be slower but can read stylized glyphs better)",
             bg=B, fg=self.DIM, font=("Helvetica", 9)).pack(side=tk.LEFT, padx=8)

        expand_frame = tk.Frame(content, bg=B)
        expand_frame.grid(row=12, column=0, sticky=tk.W, pady=(0,10))
        auto_expand_var = tk.BooleanVar(value=self.auto_expand_bbox)
        tk.Checkbutton(
            expand_frame,
            text="Auto-expand board area (helps partial hole cards)",
            variable=auto_expand_var,
            bg=B,
            fg=F,
            selectcolor=self.BG2,
            activebackground=B,
            activeforeground=F,
        ).pack(anchor=tk.W)
        
        # White ratio threshold
        tk.Label(content, text="Card Detection:", 
             bg=B, fg=F, font=("Helvetica", 11, "bold")).grid(row=13, column=0, sticky=tk.W, pady=(8,4))
        
        white_ratio_frame = tk.Frame(content, bg=B)
        white_ratio_frame.grid(row=14, column=0, sticky=tk.W, pady=(0,16))
        
        tk.Label(white_ratio_frame, text="White ratio threshold:", bg=B, fg=F).pack(side=tk.LEFT, padx=(0,4))
        white_ratio_var = tk.DoubleVar(value=self.white_ratio_threshold)
        tk.Scale(white_ratio_frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                variable=white_ratio_var, bg=B, fg=F, highlightthickness=0,
                troughcolor=self.BG2, length=150).pack(side=tk.LEFT, padx=2)
        tk.Label(white_ratio_frame, text="(lower = detect darker/colored cards)", 
                 bg=B, fg=self.DIM, font=("Helvetica", 9)).pack(side=tk.LEFT, padx=4)
        
        # Aspect ratio thresholds
        aspect_frame = tk.Frame(content, bg=B)
        aspect_frame.grid(row=15, column=0, sticky=tk.W, pady=(0,8))
        
        tk.Label(aspect_frame, text="Aspect ratio:", bg=B, fg=F).pack(side=tk.LEFT, padx=(0,4))
        aspect_min_var = tk.DoubleVar(value=self.aspect_ratio_min)
        tk.Scale(aspect_frame, from_=0.3, to=0.9, resolution=0.05, orient=tk.HORIZONTAL,
                variable=aspect_min_var, bg=B, fg=F, highlightthickness=0,
                troughcolor=self.BG2, length=100).pack(side=tk.LEFT, padx=2)
        tk.Label(aspect_frame, text="to", bg=B, fg=F).pack(side=tk.LEFT, padx=4)
        aspect_max_var = tk.DoubleVar(value=self.aspect_ratio_max)
        tk.Scale(aspect_frame, from_=0.5, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                variable=aspect_max_var, bg=B, fg=F, highlightthickness=0,
                troughcolor=self.BG2, length=100).pack(side=tk.LEFT, padx=2)
        tk.Label(aspect_frame, text="(short/long)", 
                 bg=B, fg=self.DIM, font=("Helvetica", 9)).pack(side=tk.LEFT, padx=4)
        
        # Fill ratio threshold
        fill_ratio_frame = tk.Frame(content, bg=B)
        fill_ratio_frame.grid(row=16, column=0, sticky=tk.W, pady=(0,8))
        
        tk.Label(fill_ratio_frame, text="Fill ratio threshold:", bg=B, fg=F).pack(side=tk.LEFT, padx=(0,4))
        fill_ratio_var = tk.DoubleVar(value=self.fill_ratio_threshold)
        tk.Scale(fill_ratio_frame, from_=0.1, to=0.9, resolution=0.05, orient=tk.HORIZONTAL,
                variable=fill_ratio_var, bg=B, fg=F, highlightthickness=0,
                troughcolor=self.BG2, length=150).pack(side=tk.LEFT, padx=2)
        tk.Label(fill_ratio_frame, text="(area/bounding_box)", 
                 bg=B, fg=self.DIM, font=("Helvetica", 9)).pack(side=tk.LEFT, padx=4)
        
        # Red ratio threshold
        red_ratio_frame = tk.Frame(content, bg=B)
        red_ratio_frame.grid(row=17, column=0, sticky=tk.W, pady=(0,8))
        
        tk.Label(red_ratio_frame, text="Red ratio threshold:", bg=B, fg=F).pack(side=tk.LEFT, padx=(0,4))
        red_ratio_var = tk.DoubleVar(value=self.red_ratio_threshold)
        tk.Scale(red_ratio_frame, from_=0.2, to=0.6, resolution=0.05, orient=tk.HORIZONTAL,
                variable=red_ratio_var, bg=B, fg=F, highlightthickness=0,
                troughcolor=self.BG2, length=150).pack(side=tk.LEFT, padx=2)
        tk.Label(red_ratio_frame, text="(hearts/diamonds detection)", 
                 bg=B, fg=self.DIM, font=("Helvetica", 9)).pack(side=tk.LEFT, padx=4)
        
        # HSV brightness thresholds
        hsv_frame = tk.Frame(content, bg=B)
        hsv_frame.grid(row=18, column=0, sticky=tk.W, pady=(0,16))
        
        tk.Label(hsv_frame, text="HSV brightness (V):", bg=B, fg=F).pack(side=tk.LEFT, padx=(0,4))
        hsv_v_min_var = tk.IntVar(value=self.hsv_v_min)
        tk.Scale(hsv_frame, from_=0, to=150, resolution=5, orient=tk.HORIZONTAL,
                variable=hsv_v_min_var, bg=B, fg=F, highlightthickness=0,
                troughcolor=self.BG2, length=100).pack(side=tk.LEFT, padx=2)
        tk.Label(hsv_frame, text="to", bg=B, fg=F).pack(side=tk.LEFT, padx=4)
        hsv_v_max_var = tk.IntVar(value=self.hsv_v_max)
        tk.Scale(hsv_frame, from_=150, to=255, resolution=5, orient=tk.HORIZONTAL,
                variable=hsv_v_max_var, bg=B, fg=F, highlightthickness=0,
                troughcolor=self.BG2, length=100).pack(side=tk.LEFT, padx=2)
        tk.Label(hsv_frame, text="(lower min = detect darker cards)", 
                 bg=B, fg=self.DIM, font=("Helvetica", 9)).pack(side=tk.LEFT, padx=4)

        hsv_sat_frame = tk.Frame(content, bg=B)
        hsv_sat_frame.grid(row=19, column=0, sticky=tk.W, pady=(0,16))

        tk.Label(hsv_sat_frame, text="HSV saturation max (S):", bg=B, fg=F).pack(side=tk.LEFT, padx=(0,4))
        hsv_s_max_var = tk.IntVar(value=self.hsv_s_max)
        tk.Scale(hsv_sat_frame, from_=0, to=255, resolution=5, orient=tk.HORIZONTAL,
            variable=hsv_s_max_var, bg=B, fg=F, highlightthickness=0,
            troughcolor=self.BG2, length=180).pack(side=tk.LEFT, padx=2)
        tk.Label(hsv_sat_frame, text="(lower = ignore colored felt and UI)", 
             bg=B, fg=self.DIM, font=("Helvetica", 9)).pack(side=tk.LEFT, padx=4)
        
        # Debug Image Previews
        tk.Label(content, text="Debug Image Previews:", 
            bg=B, fg=F, font=("Helvetica", 11, "bold")).grid(row=20, column=0, sticky=tk.W, pady=(8,4))
        tk.Label(content, text="Recent OCR processing images (check HSV Mask shows white cards, not just balance numbers)", 
            bg=B, fg=self.DIM, font=("Helvetica", 9)).grid(row=21, column=0, sticky=tk.W, pady=(0,8))
        
        preview_frame = tk.Frame(content, bg=B)
        preview_frame.grid(row=22, column=0, sticky=tk.W, pady=(0,16))
        
        # Create preview labels (will be populated with images)
        preview_labels = {}
        preview_images = {}  # Keep references to prevent garbage collection
        
        for i, (name, filename) in enumerate([
            ("Card", "card_full_0.png"),
            ("Rank", "rank_0.png"),
            ("Suit", "suit_0.png"),
            ("HSV Mask", "01a_hsv_mask_raw.png")
        ]):
            col_frame = tk.Frame(preview_frame, bg=B)
            col_frame.grid(row=0, column=i, padx=8)
            
            tk.Label(col_frame, text=name, bg=B, fg=F, font=("Helvetica", 9)).pack()
            img_label = tk.Label(col_frame, bg=self.BG2, width=20, height=14)
            img_label.pack(pady=4)
            preview_labels[filename] = img_label
        
        def refresh_previews():
            """Load and display the most recent debug images."""
            import os
            debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_crops")
            
            for filename, label in preview_labels.items():
                filepath = os.path.join(debug_dir, filename)
                try:
                    if os.path.exists(filepath):
                        img = Image.open(filepath)
                        # Resize to fit preview (maintain aspect ratio)
                        img.thumbnail((180, 180), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        preview_images[filename] = photo  # Keep reference
                        label.config(image=photo)
                    else:
                        label.config(image='', text='No image')
                except Exception as e:
                    label.config(image='', text=f'Error:\n{str(e)[:20]}')
        
        refresh_btn = tk.Button(preview_frame, text="↻ Refresh Previews", 
                               command=refresh_previews,
                               bg=self.BG2, fg=F, font=("Helvetica", 9),
                               relief=tk.RAISED, bd=1, padx=8, pady=2)
        refresh_btn.grid(row=1, column=0, columnspan=4, pady=8)
        
        # Load previews initially and set up auto-refresh
        refresh_previews()
        self.preview_refresh_callback = refresh_previews
        
        # Clear callback and unbind mousewheel when window closes
        def on_close():
            self.preview_refresh_callback = None
            canvas.unbind_all("<MouseWheel>")
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", on_close)
        
        # Buttons
        btn_frame = tk.Frame(win, bg=B)
        btn_frame.pack(pady=12)
        
        def apply():
            # Build rotation angles list
            min_ang = angle_min_var.get()
            max_ang = angle_max_var.get()
            step = angle_step_var.get()
            if min_ang <= max_ang and step > 0:
                angles = list(range(min_ang, max_ang + 1, step))
                if 0 not in angles:
                    angles.append(0)
                    angles.sort()
                self.ocr_rotation_angles = angles
                self.ocr_angle_step = step
            
            # Build orientations list
            self.ocr_orientations = [ang for ang, var in orient_vars.items() if var.get()]
            if not self.ocr_orientations:
                self.ocr_orientations = [0]  # Must have at least one
            
            self.ocr_rank_threshold = rank_thresh_var.get()
            self.ocr_suit_threshold = suit_thresh_var.get()
            self.ocr_upscale_factor = upscale_var.get()
            self.ocr_engine = ocr_engine_var.get()
            self.auto_expand_bbox = auto_expand_var.get()
            self.white_ratio_threshold = white_ratio_var.get()
            self.aspect_ratio_min = aspect_min_var.get()
            self.aspect_ratio_max = aspect_max_var.get()
            self.fill_ratio_threshold = fill_ratio_var.get()
            self.red_ratio_threshold = red_ratio_var.get()
            self.hsv_v_min = hsv_v_min_var.get()
            self.hsv_v_max = hsv_v_max_var.get()
            self.hsv_s_max = hsv_s_max_var.get()
            
            self._set_status(f"OCR params: engine={self.ocr_engine}, angles={self.ocr_rotation_angles}, orientations={self.ocr_orientations}")
            self.preview_refresh_callback = None
            canvas.unbind_all("<MouseWheel>")
            win.destroy()
        
        def reset():
            self.ocr_rotation_angles = [-5, 0, 5]
            self.ocr_angle_step = 5
            self.ocr_orientations = [0, 180]
            self.ocr_engine = "pytesseract"
            self.auto_expand_bbox = False
            self.ocr_rank_threshold = 0.60
            self.ocr_suit_threshold = 0.20
            self.ocr_upscale_factor = 2
            self.white_ratio_threshold = 0.24
            self.aspect_ratio_min = 0.45
            self.aspect_ratio_max = 0.88
            self.fill_ratio_threshold = 0.38
            self.red_ratio_threshold = 0.38
            self.hsv_v_min = 95
            self.hsv_v_max = 255
            self.hsv_s_max = 75
            self._set_status("OCR parameters reset to defaults")
            self.preview_refresh_callback = None
            canvas.unbind_all("<MouseWheel>")
            win.destroy()
        
        def cancel():
            self.preview_refresh_callback = None
            canvas.unbind_all("<MouseWheel>")
            win.destroy()
        
        self._btn(btn_frame, "Apply", apply).pack(side=tk.LEFT, padx=6)
        self._btn(btn_frame, "Reset to Defaults", reset).pack(side=tk.LEFT, padx=6)
        self._btn(btn_frame, "Cancel", cancel).pack(side=tk.LEFT, padx=6)
    
    def _set_board_area(self):
        """Open transparent overlay to select screen capture region."""
        # Minimize main window to get clean screenshot
        self.iconify()
        self.update()
        
        # Wait a moment for window to minimize
        import time
        time.sleep(0.2)
        
        # Capture current screen
        screenshot = ImageGrab.grab()
        
        # DEBUG: Check coordinate system
        import sys
        print(f"[DEBUG] Full screenshot size: {screenshot.width}x{screenshot.height}", file=sys.stderr, flush=True)
        
        # Create fullscreen overlay window
        overlay = tk.Toplevel(self)
        overlay.attributes('-fullscreen', True)
        overlay.attributes('-topmost', True)
        overlay.configure(bg='black')
        
        # Get logical screen dimensions
        screen_w = overlay.winfo_screenwidth()
        screen_h = overlay.winfo_screenheight()
        
        print(f"[DEBUG] Logical screen size: {screen_w}x{screen_h}", file=sys.stderr, flush=True)
        
        # Calculate scale factor for HiDPI displays (Retina, etc.)
        scale_x = screenshot.width / screen_w
        scale_y = screenshot.height / screen_h
        
        print(f"[DEBUG] Scale factors: {scale_x:.2f}x{scale_y:.2f}", file=sys.stderr, flush=True)
        
        # Scale screenshot to match logical screen size
        if screenshot.width != screen_w or screenshot.height != screen_h:
            screenshot_display = screenshot.resize((screen_w, screen_h), Image.LANCZOS)
        else:
            screenshot_display = screenshot
        
        # Canvas for drawing
        canvas = tk.Canvas(overlay, highlightthickness=0, width=screen_w, height=screen_h)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Display screenshot as background
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(screenshot_display)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.image = photo  # Keep reference
        
        # Initial rectangle (center of screen)
        rect_w, rect_h = 800, 600
        rect_x1 = (screen_w - rect_w) // 2
        rect_y1 = (screen_h - rect_h) // 2
        rect_x2 = rect_x1 + rect_w
        rect_y2 = rect_y1 + rect_h
        
        # Draw red rectangle with 5px border
        rect = canvas.create_rectangle(rect_x1, rect_y1, rect_x2, rect_y2,
                                       outline='red', width=5, fill='')
        
        # Instructions with semi-transparent background
        text_bg = canvas.create_rectangle(0, 0, screen_w, 80, fill='black', stipple='gray50')
        info_text = canvas.create_text(screen_w // 2, 30,
                                       text="Drag to move/resize | F = fullscreen | ENTER = confirm | ESC = cancel | DELETE = reset",
                                       fill='yellow', font=('Helvetica', 14, 'bold'))
        coords_text = canvas.create_text(screen_w // 2, 60,
                                         text=f"Selection: {int(rect_w * scale_x)}x{int(rect_h * scale_y)} pixels (native resolution)",
                                         fill='white', font=('Helvetica', 12))
        
        # State for dragging
        drag_data = {'item': None, 'x': 0, 'y': 0, 'mode': None}
        
        def get_cursor_mode(x, y):
            """Determine if cursor is on edge/corner for resizing."""
            coords = canvas.coords(rect)
            if not coords:
                return None
            x1, y1, x2, y2 = coords
            margin = 15
            
            on_left = abs(x - x1) < margin
            on_right = abs(x - x2) < margin
            on_top = abs(y - y1) < margin
            on_bottom = abs(y - y2) < margin
            
            if on_top and on_left:
                return 'nw'
            elif on_top and on_right:
                return 'ne'
            elif on_bottom and on_left:
                return 'sw'
            elif on_bottom and on_right:
                return 'se'
            elif on_left:
                return 'w'
            elif on_right:
                return 'e'
            elif on_top:
                return 'n'
            elif on_bottom:
                return 's'
            elif x1 < x < x2 and y1 < y < y2:
                return 'move'
            return None
        
        def on_press(event):
            drag_data['x'] = event.x
            drag_data['y'] = event.y
            drag_data['mode'] = get_cursor_mode(event.x, event.y)
        
        def on_drag(event):
            if drag_data['mode'] is None:
                return
            
            dx = event.x - drag_data['x']
            dy = event.y - drag_data['y']
            coords = canvas.coords(rect)
            x1, y1, x2, y2 = coords
            
            mode = drag_data['mode']
            
            # Calculate new coordinates based on mode
            if mode == 'move':
                # Move entire rectangle
                new_x1, new_y1, new_x2, new_y2 = x1 + dx, y1 + dy, x2 + dx, y2 + dy
            elif mode == 'nw':
                new_x1, new_y1, new_x2, new_y2 = x1 + dx, y1 + dy, x2, y2
            elif mode == 'ne':
                new_x1, new_y1, new_x2, new_y2 = x1, y1 + dy, x2 + dx, y2
            elif mode == 'sw':
                new_x1, new_y1, new_x2, new_y2 = x1 + dx, y1, x2, y2 + dy
            elif mode == 'se':
                new_x1, new_y1, new_x2, new_y2 = x1, y1, x2 + dx, y2 + dy
            elif mode == 'n':
                new_x1, new_y1, new_x2, new_y2 = x1, y1 + dy, x2, y2
            elif mode == 's':
                new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2 + dy
            elif mode == 'w':
                new_x1, new_y1, new_x2, new_y2 = x1 + dx, y1, x2, y2
            elif mode == 'e':
                new_x1, new_y1, new_x2, new_y2 = x1, y1, x2 + dx, y2
            else:
                return
            
            # Clamp to screen bounds (allow full screen selection)
            if mode == 'move':
                # Special handling for move: keep rectangle together
                width = new_x2 - new_x1
                height = new_y2 - new_y1
                # Ensure rectangle stays fully on screen
                if new_x1 < 0:
                    new_x1 = 0
                    new_x2 = width
                if new_x2 > screen_w:
                    new_x2 = screen_w
                    new_x1 = screen_w - width
                if new_y1 < 0:
                    new_y1 = 0
                    new_y2 = height
                if new_y2 > screen_h:
                    new_y2 = screen_h
                    new_y1 = screen_h - height
            else:
                # For resize operations: allow coordinates to swap and clamp independently
                new_x1 = max(0, min(screen_w, new_x1))
                new_y1 = max(0, min(screen_h, new_y1))
                new_x2 = max(0, min(screen_w, new_x2))
                new_y2 = max(0, min(screen_h, new_y2))
            
            # Ensure minimum size of 50x50
            if abs(new_x2 - new_x1) >= 50 and abs(new_y2 - new_y1) >= 50:
                canvas.coords(rect, new_x1, new_y1, new_x2, new_y2)
                # Update coordinates display
                w = int(abs(new_x2 - new_x1) * scale_x)
                h = int(abs(new_y2 - new_y1) * scale_y)
                canvas.itemconfig(coords_text, text=f"Selection: {w}x{h} pixels (native resolution)")
            
            drag_data['x'] = event.x
            drag_data['y'] = event.y
        
        def on_motion(event):
            """Update cursor based on position."""
            mode = get_cursor_mode(event.x, event.y)
            if mode == 'move':
                canvas.config(cursor='fleur')
            elif mode in ('nw', 'se'):
                canvas.config(cursor='sizing')
            elif mode in ('ne', 'sw'):
                canvas.config(cursor='sizing')
            elif mode in ('n', 's'):
                canvas.config(cursor='sb_v_double_arrow')
            elif mode in ('w', 'e'):
                canvas.config(cursor='sb_h_double_arrow')
            else:
                canvas.config(cursor='arrow')
        
        def confirm(event=None):
            coords = canvas.coords(rect)
            if coords:
                x1, y1, x2, y2 = coords
                import sys
                print(f"[BOARD-AREA] Logical coords: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})", file=sys.stderr, flush=True)
                print(f"[BOARD-AREA] Scale factors: scale_x={scale_x:.2f}, scale_y={scale_y:.2f}", file=sys.stderr, flush=True)
                print(f"[BOARD-AREA] Screen: logical={screen_w}x{screen_h}, screenshot={screenshot.width}x{screenshot.height}", file=sys.stderr, flush=True)
                
                # Store NATIVE (scaled) coordinates for macOS ImageGrab manual cropping
                self.capture_bbox = (
                    int(min(x1, x2) * scale_x),
                    int(min(y1, y2) * scale_y),
                    int(max(x1, x2) * scale_x),
                    int(max(y1, y2) * scale_y)
                )
                print(f"[BOARD-AREA] Native bbox: {self.capture_bbox}", file=sys.stderr, flush=True)
                
                w = int((max(x1, x2) - min(x1, x2)))
                h = int((max(y1, y2) - min(y1, y2)))
                self.board_area_btn.config(text=f"Board Area: {w}x{h}")
                self._set_status(f"Board area set: {w}x{h} pixels (logical)")
            overlay.destroy()
            self.deiconify()
        
        def cancel(event=None):
            overlay.destroy()
            self.deiconify()
        
        def reset_area(event=None):
            self.capture_bbox = None
            self.board_area_btn.config(text="Set Board Area")
            self._set_status("Board area reset to full screen")
            overlay.destroy()
            self.deiconify()
        
        def maximize_selection(event=None):
            """Expand selection to full screen."""
            canvas.coords(rect, 0, 0, screen_w, screen_h)
            w = int(screen_w * scale_x)
            h = int(screen_h * scale_y)
            canvas.itemconfig(coords_text, text=f"Selection: {w}x{h} pixels (FULL SCREEN)")
        
        canvas.bind('<ButtonPress-1>', on_press)
        canvas.bind('<B1-Motion>', on_drag)
        canvas.bind('<Motion>', on_motion)
        overlay.bind('<Return>', confirm)
        overlay.bind('<Escape>', cancel)
        overlay.bind('<Delete>', reset_area)
        overlay.bind('<BackSpace>', reset_area)
        overlay.bind('f', maximize_selection)
        overlay.bind('F', maximize_selection)
        overlay.bind('<BackSpace>', reset_area)
        overlay.focus_set()

    def _set_card_angles(self):
        """Open transparent overlay to set rotation angles for hole cards with visual green lines."""
        import math
        
        # Minimize main window to get clean screenshot
        self.iconify()
        self.update()
        
        # Wait a moment for window to minimize
        import time
        time.sleep(0.2)
        
        # Capture current screen
        screenshot = ImageGrab.grab()
        
        # Create fullscreen overlay window
        overlay = tk.Toplevel(self)
        overlay.attributes('-fullscreen', True)
        overlay.attributes('-topmost', True)
        overlay.configure(bg='black')
        
        # Get logical screen dimensions
        screen_w = overlay.winfo_screenwidth()
        screen_h = overlay.winfo_screenheight()
        
        # Scale screenshot to match logical screen size
        if screenshot.width != screen_w or screenshot.height != screen_h:
            screenshot_display = screenshot.resize((screen_w, screen_h), Image.LANCZOS)
        else:
            screenshot_display = screenshot
        
        # Canvas for drawing
        canvas = tk.Canvas(overlay, highlightthickness=0, width=screen_w, height=screen_h)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Display screenshot as background
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(screenshot_display)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.image = photo  # Keep reference
        
        # Card positions (left 1/3 and right 1/3 of screen, vertically centered)
        card1_center = (screen_w // 3, screen_h // 2)
        card2_center = (2 * screen_w // 3, screen_h // 2)
        line_length = 150
        
        # Convert current rotation angles to radians (0° = pointing right, positive = counterclockwise)
        angle1_deg = self.card1_rotation
        angle2_deg = self.card2_rotation
        
        def deg_to_rad(deg):
            return math.radians(deg)
        
        def rad_to_deg(rad):
            return math.degrees(rad)
        
        def calc_line_end(center_x, center_y, angle_deg, length):
            """Calculate line endpoint from center, angle (degrees), and length."""
            angle_rad = deg_to_rad(angle_deg)
            end_x = center_x + length * math.cos(angle_rad)
            end_y = center_y - length * math.sin(angle_rad)  # Negative because y increases downward
            return end_x, end_y
        
        # Draw card 1 line (green)
        end1_x, end1_y = calc_line_end(card1_center[0], card1_center[1], angle1_deg, line_length)
        line1 = canvas.create_line(card1_center[0], card1_center[1], end1_x, end1_y,
                                   fill='#00ff00', width=5, arrow=tk.LAST, arrowshape=(16, 20, 8))
        circle1 = canvas.create_oval(card1_center[0]-8, card1_center[1]-8,
                                     card1_center[0]+8, card1_center[1]+8,
                                     fill='#00ff00', outline='white', width=2)
        label1 = canvas.create_text(card1_center[0], card1_center[1] - 180,
                                    text=f"CARD 1: {angle1_deg}°",
                                    fill='#00ff00', font=('Helvetica', 16, 'bold'))
        
        # Draw card 2 line (green)
        end2_x, end2_y = calc_line_end(card2_center[0], card2_center[1], angle2_deg, line_length)
        line2 = canvas.create_line(card2_center[0], card2_center[1], end2_x, end2_y,
                                   fill='#00ff00', width=5, arrow=tk.LAST, arrowshape=(16, 20, 8))
        circle2 = canvas.create_oval(card2_center[0]-8, card2_center[1]-8,
                                     card2_center[0]+8, card2_center[1]+8,
                                     fill='#00ff00', outline='white', width=2)
        label2 = canvas.create_text(card2_center[0], card2_center[1] - 180,
                                    text=f"CARD 2: {angle2_deg}°",
                                    fill='#00ff00', font=('Helvetica', 16, 'bold'))
        
        # Instructions
        text_bg = canvas.create_rectangle(0, 0, screen_w, 100, fill='black', stipple='gray50')
        info_text = canvas.create_text(screen_w // 2, 30,
                                       text="Drag the green arrows to set card rotation angles",
                                       fill='yellow', font=('Helvetica', 16, 'bold'))
        hint_text = canvas.create_text(screen_w // 2, 60,
                                       text="ENTER = confirm | ESC = cancel | R = reset to 0°",
                                       fill='white', font=('Helvetica', 14))
        
        # State for dragging
        drag_state = {'active_line': None}
        
        def get_angle(center_x, center_y, mouse_x, mouse_y):
            """Calculate angle from center to mouse position (in degrees)."""
            dx = mouse_x - center_x
            dy = center_y - mouse_y  # Flip y because canvas y increases downward
            angle_rad = math.atan2(dy, dx)
            angle_deg = rad_to_deg(angle_rad)
            # Round to nearest 5 degrees
            angle_deg = round(angle_deg / 5) * 5
            # Normalize to -180 to 180
            while angle_deg > 180:
                angle_deg -= 360
            while angle_deg < -180:
                angle_deg += 360
            return angle_deg
        
        def update_line(line_num):
            """Redraw the specified line with current angle."""
            nonlocal angle1_deg, angle2_deg
            
            if line_num == 1:
                center = card1_center
                angle = angle1_deg
                line = line1
                label = label1
                label_text = f"CARD 1: {angle}°"
            else:
                center = card2_center
                angle = angle2_deg
                line = line2
                label = label2
                label_text = f"CARD 2: {angle}°"
            
            end_x, end_y = calc_line_end(center[0], center[1], angle, line_length)
            canvas.coords(line, center[0], center[1], end_x, end_y)
            canvas.itemconfig(label, text=label_text)
        
        def on_press(event):
            """Check if user clicked near a line endpoint."""
            # Check distance to card 1 endpoint
            end1_x, end1_y = calc_line_end(card1_center[0], card1_center[1], angle1_deg, line_length)
            dist1 = math.sqrt((event.x - end1_x)**2 + (event.y - end1_y)**2)
            
            # Check distance to card 2 endpoint
            end2_x, end2_y = calc_line_end(card2_center[0], card2_center[1], angle2_deg, line_length)
            dist2 = math.sqrt((event.x - end2_x)**2 + (event.y - end2_y)**2)
            
            # Activate the closest line if within 50 pixels
            if dist1 < 50:
                drag_state['active_line'] = 1
                canvas.itemconfig(line1, fill='#ffff00', width=6)  # Highlight
            elif dist2 < 50:
                drag_state['active_line'] = 2
                canvas.itemconfig(line2, fill='#ffff00', width=6)  # Highlight
        
        def on_drag(event):
            """Update the active line angle as mouse moves."""
            nonlocal angle1_deg, angle2_deg
            
            if drag_state['active_line'] == 1:
                angle1_deg = get_angle(card1_center[0], card1_center[1], event.x, event.y)
                update_line(1)
            elif drag_state['active_line'] == 2:
                angle2_deg = get_angle(card2_center[0], card2_center[1], event.x, event.y)
                update_line(2)
        
        def on_release(event):
            """Deactivate line when mouse released."""
            if drag_state['active_line'] == 1:
                canvas.itemconfig(line1, fill='#00ff00', width=5)
            elif drag_state['active_line'] == 2:
                canvas.itemconfig(line2, fill='#00ff00', width=5)
            drag_state['active_line'] = None
        
        def confirm(event=None):
            """Save angles and close overlay."""
            self.card1_rotation = angle1_deg
            self.card2_rotation = angle2_deg
            self.hole_rotation_var.set(angle1_deg)  # For backward compatibility
            self.board_rotation_var.set(0)  # Board cards stay at 0
            
            angle_text = f"Card 1: {angle1_deg}°, Card 2: {angle2_deg}°"
            self.card_angles_btn.config(text=f"Card Angles: {angle_text}")
            self._set_status(f"Card angles set: {angle_text}")
            overlay.destroy()
            self.deiconify()
        
        def cancel(event=None):
            overlay.destroy()
            self.deiconify()
        
        def reset_angles(event=None):
            """Reset both angles to 0 (auto-detect mode)."""
            nonlocal angle1_deg, angle2_deg
            angle1_deg = 0
            angle2_deg = 0
            update_line(1)
            update_line(2)
            # Also reset the stored values
            self.card1_rotation = 0
            self.card2_rotation = 0
            self.card_angles_btn.config(text="Set Card Angles (Auto-detect)")
        
        # Bind events
        canvas.bind('<ButtonPress-1>', on_press)
        canvas.bind('<B1-Motion>', on_drag)
        canvas.bind('<ButtonRelease-1>', on_release)
        overlay.bind('<Return>', confirm)
        overlay.bind('<Escape>', cancel)
        overlay.bind('r', reset_angles)
        overlay.bind('R', reset_angles)
        overlay.focus_set()

    def _on_capture(self):
        self._set_status("Capturing screen…")
        self.ocr_text.delete("1.0", tk.END)
        self.ocr_text.insert(tk.END, "Capturing screen and running OCR...\n")
        self._set_pending_result("Analyzing…", "Waiting for OCR results from the screen capture.")
        self.capture_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._capture_worker, daemon=True).start()

    def _capture_worker(self):
        try:
            raw_text, cards, angles = capture_and_ocr(
                self.ocr_rotation_angles, 
                self.ocr_orientations,
                self.ocr_upscale_factor,
                self.ocr_rank_threshold,
                self.ocr_suit_threshold,
                self.white_ratio_threshold,
                self.aspect_ratio_min,
                self.aspect_ratio_max,
                self.fill_ratio_threshold,
                self.red_ratio_threshold,
                self.hole_rotation_var.get(),
                self.board_rotation_var.get(),
                self.card1_rotation,
                self.card2_rotation,
                self.hsv_v_min,
                self.hsv_v_max,
                self.hsv_s_max,
                self.ocr_engine,
                self.auto_expand_bbox,
                self.capture_bbox
            )
            self.detected_card_angles = angles  # Store for display
            self.after(0, lambda: self._display_ocr(raw_text, cards, angles))
            self.after(0, lambda: self._run_analysis(cards))
            # Auto-refresh preview window if open
            if self.preview_refresh_callback:
                self.after(0, self.preview_refresh_callback)
        except Exception as exc:
            self.after(0, lambda: self._show_result("Capture failed", str(exc)))
            self.after(0, lambda: messagebox.showerror("Error", str(exc)))
            self.after(0, lambda: self._set_status("Error during capture."))
        finally:
            self.after(0, lambda: self.capture_btn.config(state=tk.NORMAL))

    def _on_manual(self):
        hole_raw = self.hole_entry.get().strip()
        board_raw = self.board_entry.get().strip()
        combined = (hole_raw + " " + board_raw).upper()
        cards = parse_cards_from_text(combined)
        self._display_ocr("(manual input)", cards)
        self._run_analysis(cards)

    def _display_ocr(self, raw_text: str, cards: List[str], angles: List[int] = None):
        self.ocr_text.delete("1.0", tk.END)
        self.ocr_text.insert(tk.END,
            f"Detected cards: {', '.join(cards) if cards else 'none'}\n")
        if angles:
            angles_str = ", ".join([f"{card}@{angle}°" for card, angle in zip(cards, angles)])
            self.ocr_text.insert(tk.END, f"Card angles: {angles_str}\n")
        preview = raw_text[:400] + ("…" if len(raw_text) > 400 else "")
        self.ocr_text.insert(tk.END, f"[OCR] {preview}")

    def _run_analysis(self, cards: List[str]):
        if len(cards) < 2:
            self._show_result("Not enough cards detected",
                              "Enter at least 2 hole cards manually or capture screen.")
            self._set_status("Need at least 2 cards.")
            return

        hole  = cards[:2]
        board = cards[2:7]

        street_sel = self.street_var.get()
        if street_sel == "Auto-detect":
            street = ("Pre-Flop" if not board else
                      "Flop"     if len(board) <= 3 else
                      "Turn"     if len(board) == 4 else "River")
        else:
            street = street_sel
        
        # Pre-flop uses only hole cards for equity simulation
        if street == "Pre-Flop":
            board = []

        position = self.position_var.get()
        num_opp  = self.opponents_var.get()

        pot_odds = None
        try:
            pot_odds = float(self.pot_odds_var.get().strip())
        except ValueError:
            pass

        self._set_status("Running Monte Carlo equity simulation…")
        self._set_pending_result("Analyzing…", "Calculating equity and generating strategy advice.")

        def worker():
            try:
                equity = monte_carlo_equity(hole, board,
                                            num_opponents=num_opp, iterations=2000)
                advice = gto_advice(hole, board, equity, street, position, pot_odds)
                self.after(0, lambda: self._show_advice(advice, hole, board, street))
                self.after(0, lambda: self._set_status("Analysis complete."))
            except Exception as exc:
                self.after(0, lambda: self._show_result("Analysis failed", str(exc)))
                self.after(0, lambda: self._set_status("Analysis failed."))

        threading.Thread(target=worker, daemon=True).start()

    def _show_advice(self, advice: dict, hole: List[str],
                     board: List[str], street: str):
        action    = advice.get("action", "—")
        equity    = advice.get("equity", 0.0)
        reasoning = advice.get("reasoning", "")
        hand_made = advice.get("hand_made", "")
        threshold = advice.get("pot_odds_threshold")

        self.action_label.config(text=f"▶  {action}")

        lines = [
            f"Street      : {street}",
            f"Hole cards  : {' '.join(hole)}",
            f"Board       : {' '.join(board) if board else '(none)'}",
            f"Best hand   : {hand_made}",
            f"Win equity  : {equity:.1%}",
        ]
        if threshold:
            lines.append(f"Pot odds req: {threshold}")
        lines += ["", f"ACTION: {action}", "", "Reasoning:", f"  {reasoning}",
                  "", "─" * 50,
                  "Equity estimated via Monte Carlo (2 000 iterations).",
                  "GTO-informed suggestions, not guarantees."]

        self.strategy_text.delete("1.0", tk.END)
        self.strategy_text.insert(tk.END, "\n".join(lines))

    def _show_result(self, action: str, details: str):
        self.action_label.config(text=action)
        self.strategy_text.delete("1.0", tk.END)
        self.strategy_text.insert(tk.END, details)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    app = PokerGeniusApp()
    app.mainloop()
