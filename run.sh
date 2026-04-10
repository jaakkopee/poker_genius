#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TK_SILENCE_DEPRECATION=1

if [[ ! -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
	echo "Missing virtual environment at $SCRIPT_DIR/.venv"
	echo "Run 'bash install_macos.sh' to create it with the Homebrew Python interpreter."
	exit 1
fi

exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/poker_genius.py" "$@"
