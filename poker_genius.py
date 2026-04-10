"""
Poker Genius - Screen capture + OCR + GTO poker advisor
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import re
from itertools import combinations
from typing import Optional, List

import pytesseract
from PIL import Image, ImageGrab, ImageEnhance, ImageFilter
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


def preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    """Enhance image for better OCR accuracy on card text."""
    img = pil_image.convert("L")
    img = img.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.5)
    img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
    return img


def parse_cards_from_text(text: str) -> List[str]:
    """
    Extract card tokens like 'Ah', 'Kd', '2s', 'Tc' from OCR output.
    Returns a list of canonical card strings e.g. ['Ah', 'Kd'].
    """
    text = text.upper().replace("♣", "C").replace("♦", "D").replace("♥", "H").replace("♠", "S")
    # Normalise common OCR confusions
    text = text.replace("0", "O").replace("1O", "TO").replace("IO", "TO")

    # Match patterns like: A H, K D, 10H, TH, 2 S ...
    pattern = re.compile(r'\b(10|[2-9TJQKA])\s*([CDHS])\b', re.IGNORECASE)
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


def capture_and_ocr() -> str:
    """Capture full screen and return raw OCR text."""
    screenshot = ImageGrab.grab()
    processed = preprocess_for_ocr(screenshot)
    text = pytesseract.image_to_string(processed, config="--psm 6")
    return text


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
            raw_text = capture_and_ocr()
            cards = parse_cards_from_text(raw_text)
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
