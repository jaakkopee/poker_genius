#!/usr/bin/env python3
"""Quick test to see what template matching returns for the saved debug crops."""

from PIL import Image
import numpy as np
import cv2
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poker_genius import template_match_symbol, get_rank_templates, get_suit_templates

def test_crop(crop_path, templates, debug_label):
    """Test template matching on a specific crop."""
    print(f"\nTesting {crop_path}:", file=sys.stderr)
    img = Image.open(crop_path)
    result, score = template_match_symbol(img, templates, debug_label=debug_label)
    print(f"  Result: {result} (score: {score:.3f})", file=sys.stderr)
    return result, score

if __name__ == "__main__":
    debug_dir = os.path.join(os.path.dirname(__file__), "debug_crops")
    
    print("\n" + "="*60, file=sys.stderr)
    print("TEMPLATE MATCHING TEST", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    # Test rank crops
    for orient in [0, 180]:
        rank_path = os.path.join(debug_dir, f"rank_{orient}.png")
        if os.path.exists(rank_path):
            test_crop(rank_path, get_rank_templates(), f"RANK@{orient}°")
    
    # Test suit crops
    for orient in [0, 180]:
        suit_path = os.path.join(debug_dir, f"suit_{orient}.png")
        if os.path.exists(suit_path):
            test_crop(suit_path, get_suit_templates(), f"SUIT@{orient}°")
    
    print("\n" + "="*60, file=sys.stderr)
