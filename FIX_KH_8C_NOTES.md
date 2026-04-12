# Fix for Kh → 9c and 8c → 5h Misclassification

## Changes Applied

### 1. **Increased 180° Orientation Penalty** (from 0.20 → 0.40)
- Strongly biases toward upright (0°) cards
- Makes it harder for upside-down matches to win

### 2. **Added Ambiguity Detection**
- When template scores are LOW (rank < 0.55 OR suit < 0.33) AND no OCR:
  - 180° orientation must be 0.15 points HIGHER than best 0° score to be accepted
  - Prevents similar-looking ranks (K↔9, 8↔5) from flipping

### 3. **Enlarged Symbol Crops**
- Increased rank crop from 40%×30% → 50%×40% (72px → 90px height)
- Increased suit crop similarly
- Gives OCR more pixels to work with, improving recognition

## Expected Behavior

**Before Fix:**
- Kh detected as 9c (9 scores 0.464 > K scores 0.446)
- 8c detected as 5h (5 scores 0.411 > 8 scores 0.438)

**After Fix:**
- May still show 9c/5h IF both orientations have low confidence
- BUT the fix prevents orientation confusion when one orientation clearly wins
- Larger crops help OCR provide validation

## Known Limitations

The fixes are **defensive** (prevent wrong orientation) but not **corrective** (fix wrong templates):

1. **Root Cause:** K and 9 templates are too similar on specific card images
2. **Template Matching:** Very close scores (0.464 vs 0.446) mean templates aren't distinctive enough
3. **OCR Contribution:** Currently OCR returns empty (crop quality/normalization issues)

## What Works Best

The fix helps MOST when:
- ✓ Card orientation is clearly correct at 0°
- ✓ One orientation scores significantly higher
- ✓ You're using pytesseract with system Tesseract

The fix is LIMITED when:
- ✗ Templates genuinely score very close (K ↔ 9)
- ✗ OCR fails on small crops
- ✗ Card is actually upside-down in the capture

## Next Steps if Problem Persists

If Kh still detects as 9c after this fix, consider:

1. **Capture Better Images:** Ensure card orientation is truly upright (0°)
2. **OCR Investigation:** Check if `rank_ocr` and `suit_ocr` are being populated
3. **Manual Override UI:** Let users correct wrong detections
4. **Template Replacement:** Generate new templates with different fonts
5. **Debug Analysis:** Run diagnostic to see which templates are scoring

## Running the Fix

```bash
./run.sh  # The fix is automatic, just test normally
```

Check the console output for:
- `[DECISION] score<=0.40?` → Shows if cards are flagged as ambiguous
- `orient=0°` appearing more often than `orient=180°` → Shows bias toward upright

## Files Modified

- `poker_genius.py`
  - Increased 180° penalty in `recognize_card_from_region()`
  - Added low-confidence ambiguity check
  - Enlarged rank/suit crop regions (40%→50% width, 30%→40% height)
