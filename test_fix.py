#!/usr/bin/env python3
"""Test the Kh→9c and 8c→5h fix by processing previous debug crops."""

import sys
sys.path.insert(0, '/Users/jaakkop/Documents/koodii/poker_genius')

from PIL import Image
from poker_genius import recognize_card_from_region

# Test the most recent captured cards
test_images = [
    ("debug_crops/card_full_0.png", "Card 1 @ 0°"),
    ("debug_crops/card_full_180.png", "Card 1 @ 180°"),
]

print("\n" + "="*70)
print("TESTING CARD RECOGNITION WITH INCREASED 180° PENALTY (0.35)")
print("="*70)

for img_path, label in test_images:
    try:
        img = Image.open(img_path)
        card, score, debug = recognize_card_from_region(img, orientations=[0, 180])
        print(f"\n{label}")
        print(f"  Result: {card} (score={score:.3f})")
        print(f"  Debug: {debug}")
    except FileNotFoundError:
        print(f"\n{label}: File not found ({img_path})")
    except Exception as e:
        print(f"\n{label}: Error: {e}")

print("\n" + "="*70)
print("If showing 'Kh' and '8c' instead of '9c' and '5h', fix is working!")
print("="*70)
