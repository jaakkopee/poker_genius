#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TK_SILENCE_DEPRECATION=1
exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/poker_genius.py" "$@"
