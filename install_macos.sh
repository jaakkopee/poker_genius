#!/usr/bin/env bash
# ────────────────────────────────────────────────────────
#  Poker Genius – macOS installer
#  Usage: bash install_macos.sh
# ────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "╔══════════════════════════════════════════╗"
echo "║        Poker Genius – macOS Setup        ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Check Python 3 ──────────────────────────────────
echo "→ Checking for Python 3…"
if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    echo "  Found: $PY_VERSION"
else
    echo ""
    echo "  ERROR: python3 not found."
    echo "  Install Python 3 from https://www.python.org/downloads/"
    echo "  or via Homebrew:  brew install python"
    exit 1
fi

# Require Python >= 3.8
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 8 ) ]]; then
    echo "  ERROR: Python 3.8+ is required (found $PY_VERSION)."
    exit 1
fi

# ── 2. Check / install Tesseract ───────────────────────
echo ""
echo "→ Checking for Tesseract OCR…"
if command -v tesseract &>/dev/null; then
    echo "  Found: $(tesseract --version 2>&1 | head -1)"
else
    echo "  Tesseract not found."
    if command -v brew &>/dev/null; then
        echo "  Installing via Homebrew…"
        brew install tesseract
    else
        echo ""
        echo "  WARNING: Homebrew not found. Please install Tesseract manually:"
        echo "    https://github.com/tesseract-ocr/tesseract#installing-tesseract"
        echo "  Or install Homebrew first: https://brew.sh"
        echo ""
        read -rp "  Continue without Tesseract? [y/N] " answer
        if [[ ! "$answer" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# ── 3. Create virtual environment ─────────────────────
echo ""
echo "→ Creating virtual environment at .venv…"
if [[ -d "$VENV_DIR" ]]; then
    echo "  .venv already exists, skipping creation."
else
    python3 -m venv "$VENV_DIR"
    echo "  Created."
fi

# ── 4. Upgrade pip ────────────────────────────────────
echo ""
echo "→ Upgrading pip…"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip

# ── 5. Install requirements ───────────────────────────
echo ""
echo "→ Installing Python dependencies from requirements.txt…"
"$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"
echo "  Done."

# ── 6. Verify tkinter ────────────────────────────────
echo ""
echo "→ Verifying tkinter availability…"
if "$VENV_DIR/bin/python" -c "import tkinter" 2>/dev/null; then
    echo "  tkinter: OK"
else
    echo "  WARNING: tkinter not available in this Python build."
    echo "  On macOS you can install a tkinter-enabled Python via:"
    echo "    brew install python-tk"
fi

# ── 7. Create launch script ───────────────────────────
LAUNCHER="$SCRIPT_DIR/run.sh"
cat > "$LAUNCHER" << 'EOF'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/poker_genius.py" "$@"
EOF
chmod +x "$LAUNCHER"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║          Installation complete!          ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Run the app:  ./run.sh"
echo "  Or manually:  .venv/bin/python poker_genius.py"
echo ""
