"""
Poker Genius - Screen capture + OCR + GTO poker advisor
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import re
from functools import lru_cache
from itertools import combinations
from typing import Optional, List, Tuple

import pytesseract
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageGrab
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
CARD_ORIENTATIONS = (0, 180)
TEMPLATE_CONFIDENCE_THRESHOLD = 0.50
CARD_CANVAS = (180, 252)
SYMBOL_TEMPLATE_SIZE = (64, 64)
CARD_MASK_LOW = np.array([0, 0, 140], dtype=np.uint8)
CARD_MASK_HIGH = np.array([180, 105, 255], dtype=np.uint8)
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


def preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
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
    return processed.resize((processed.width * 2, processed.height * 2), Image.LANCZOS)


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


def template_match_symbol(pil_image: Image.Image, templates: dict, allowed: Optional[set] = None) -> Tuple[Optional[str], float]:
    """Return the best matching canonical symbol and its normalized correlation score."""
    normalized = normalize_symbol_patch(pil_image)
    if normalized is None:
        return None, 0.0

    best_symbol = None
    best_score = 0.0
    candidate = normalized.astype(np.float32)
    for symbol, variants in templates.items():
        if allowed and symbol not in allowed:
            continue
        for template in variants:
            score = float(cv2.matchTemplate(candidate, template.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0][0])
            if score > best_score:
                best_symbol = symbol
                best_score = score

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


def estimate_red_suit(pil_image: Image.Image) -> bool:
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
    
    # Check if foreground is predominantly dark (black suits) or light/colored
    avg_intensity = foreground.mean()
    if avg_intensity < 65:
        # Dark foreground → black suit (clubs/spades)
        return False
    
    # For lighter/colored foreground, check for red dominance
    red_dominant = (
        (foreground[:, 0] > foreground[:, 1] + 15)
        & (foreground[:, 0] > foreground[:, 2] + 15)
    ).mean()
    return bool(red_dominant > 0.25)


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


def find_card_regions(pil_image: Image.Image) -> List[Tuple[Tuple[int, int], Image.Image, Tuple[int, int, int, int]]]:
    """Locate bright, card-shaped regions and rectify them into portrait card crops."""
    rgb = np.array(pil_image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, CARD_MASK_LOW, CARD_MASK_HIGH)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = rgb.shape[0] * rgb.shape[1]
    min_area = max(2500, image_area * 0.001)
    max_area = image_area * 0.08
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (width, height), _ = rect
        if min(width, height) < 40:
            continue

        short_side, long_side = sorted((width, height))
        ratio = short_side / long_side
        fill_ratio = area / max(width * height, 1)
        if not (0.50 <= ratio <= 0.82 and fill_ratio >= 0.45):
            continue

        x, y, box_w, box_h = cv2.boundingRect(contour)
        warped = rectify_card_region(rgb, rect)
        warped_hsv = cv2.cvtColor(np.array(warped.convert("RGB")), cv2.COLOR_RGB2HSV)
        white_ratio = float((cv2.inRange(warped_hsv, CARD_MASK_LOW, CARD_MASK_HIGH) > 0).mean())
        if white_ratio < 0.55:
            continue
        candidates.append(((int(center_x), int(center_y)), warped, (x, y, box_w, box_h), area))

    candidates.sort(key=lambda item: item[3], reverse=True)
    deduped = []
    for center, warped, bbox, area in candidates:
        if any(abs(center[0] - prev_center[0]) < 35 and abs(center[1] - prev_center[1]) < 35 for prev_center, _, _, _ in deduped):
            continue
        deduped.append((center, warped, bbox, area))

    deduped.sort(key=lambda item: (item[0][1], item[0][0]))
    return [(center, warped, bbox) for center, warped, bbox, _ in deduped]


def rotate_for_ocr(pil_image: Image.Image, angle: int) -> Image.Image:
    """Rotate image to help OCR recover tilted ranks and suits."""
    if angle == 0:
        return pil_image
    return pil_image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))


def recognize_card_from_region(card_image: Image.Image) -> Tuple[Optional[str], float, str]:
    """Recognize a card from a rectified crop using template matching with OCR fallback."""
    best_card = None
    best_score = 0.0
    best_debug = ""

    for orientation in CARD_ORIENTATIONS:
        oriented = rotate_for_ocr(card_image, orientation)
        if oriented.width > oriented.height:
            oriented = oriented.rotate(90, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        oriented = oriented.resize(CARD_CANVAS, Image.LANCZOS)

        corner = oriented.crop((0, 0, int(oriented.width * 0.36), int(oriented.height * 0.31)))
        rank_crop = corner.crop((0, 0, int(corner.width * 0.58), int(corner.height * 0.52)))
        suit_crop = corner.crop((0, int(corner.height * 0.38), int(corner.width * 0.68), int(corner.height * 0.95)))

        rank_match, rank_score = template_match_symbol(rank_crop, get_rank_templates())
        # Skip OCR if template matching is confident
        rank_ocr = None
        if rank_score < TEMPLATE_CONFIDENCE_THRESHOLD:
            rank_ocr = normalize_rank_symbol(ocr_single_symbol(rank_crop, "0123456789TJQKAIO"))
        rank = rank_match
        if rank_ocr and (rank_ocr == rank_match or rank_score < 0.38):
            rank = rank_ocr

        allowed_suits = RED_SUITS if estimate_red_suit(suit_crop) else BLACK_SUITS
        suit_match, suit_score = template_match_symbol(suit_crop, get_suit_templates(), allowed_suits)
        # Skip OCR if template matching is confident
        suit_ocr = None
        if suit_score < TEMPLATE_CONFIDENCE_THRESHOLD:
            suit_ocr = normalize_suit_symbol(ocr_single_symbol(suit_crop, "CDHScdhs♣♦♥♠"))
        suit = suit_match
        if suit_ocr and suit_ocr in allowed_suits and (suit_ocr == suit_match or suit_score < 0.28):
            suit = suit_ocr

        if rank and suit:
            score = rank_score + suit_score
            if score > best_score:
                best_card = rank + suit
                best_score = score
                best_debug = (
                    f"orientation={orientation} rank={rank}({rank_score:.2f}/{rank_ocr or '-'}) "
                    f"suit={suit}({suit_score:.2f}/{suit_ocr or '-'})"
                )
                # Early exit if we have high confidence
                if score > 1.5:
                    break

    return best_card, best_score, best_debug


def detect_cards_by_regions(pil_image: Image.Image) -> Tuple[List[str], List[str]]:
    """Detect cards from likely card regions using template matching and region-local OCR."""
    cards = []
    debug_lines = []
    detections = []

    for index, (center, card_image, bbox) in enumerate(find_card_regions(pil_image), start=1):
        card, score, debug = recognize_card_from_region(card_image)
        if not card:
            debug_lines.append(f"region {index} @ {center}: no card match")
            continue
        detections.append((center, card, score, bbox, debug))

    detections.sort(key=lambda item: item[2], reverse=True)
    if len(detections) > 7:
        detections = detections[:7]
    detections.sort(key=lambda item: (item[0][1], item[0][0]))

    for center, card, score, bbox, debug in detections:
        if card not in cards:
            cards.append(card)
        debug_lines.append(f"region {center} bbox={bbox} -> {card} score={score:.2f} {debug}")

    return cards, debug_lines


def ocr_cards_from_image(pil_image: Image.Image) -> Tuple[str, List[str]]:
    """Use region detection first, then multi-angle OCR as a fallback and supplement."""
    region_cards, region_debug = detect_cards_by_regions(pil_image)
    all_cards = list(region_cards)
    
    # Skip expensive full-image OCR if region detection found enough cards
    if len(region_cards) >= 2:
        preview_lines = [f"Region detector found: {', '.join(region_cards)}"]
        preview_lines.extend(region_debug[:8])
        preview_lines.append("Skipped full-image OCR (sufficient cards from regions)")
        return "\n".join(preview_lines), region_cards
    
    best_cards: List[str] = []
    best_text = ""
    best_angle = 0
    best_score = (-1, 0)

    for angle in OCR_ROTATION_ANGLES:
        rotated = rotate_for_ocr(pil_image, angle)
        processed = preprocess_for_ocr(rotated)
        text = pytesseract.image_to_string(processed, config=OCR_CONFIG)
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

    return "\n".join(preview_lines) + "\n" + best_text, merged_cards


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


def capture_and_ocr() -> Tuple[str, List[str]]:
    """Capture full screen, OCR multiple tilt angles, and return the best text plus merged cards."""
    screenshot = ImageGrab.grab()
    return ocr_cards_from_image(screenshot)


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
        return tk.Label(parent, text=text,
                        bg=self.BG, fg=self.FG,
                        font=("Helvetica", 11), **kw)

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
        self.geometry("920x720")
        self.minsize(720, 600)
        self.configure(bg=self.BG)
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

        # ── Buttons ────────────────────────────────
        bf = tk.Frame(self, bg=B)
        bf.pack(pady=6)
        self.capture_btn = self._btn(bf, "Capture Screen & Analyze", self._on_capture)
        self.capture_btn.pack(side=tk.LEFT, padx=6)
        self._btn(bf, "Analyze Manual Input", self._on_manual).pack(side=tk.LEFT, padx=6)
        self._btn(bf, "Clear", self._on_clear).pack(side=tk.LEFT, padx=6)

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

    def _on_capture(self):
        self._set_status("Capturing screen…")
        self.ocr_text.delete("1.0", tk.END)
        self.ocr_text.insert(tk.END, "Capturing screen and running OCR...\n")
        self._set_pending_result("Analyzing…", "Waiting for OCR results from the screen capture.")
        self.capture_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._capture_worker, daemon=True).start()

    def _capture_worker(self):
        try:
            raw_text, cards = capture_and_ocr()
            self.after(0, lambda: self._display_ocr(raw_text, cards))
            self.after(0, lambda: self._run_analysis(cards))
        except Exception as exc:
            self.after(0, lambda: self._display_ocr(f"Capture error: {exc}", []))
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

    def _display_ocr(self, raw_text: str, cards: List[str]):
        self.ocr_text.delete("1.0", tk.END)
        self.ocr_text.insert(tk.END,
            f"Detected cards: {', '.join(cards) if cards else 'none'}\n")
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
