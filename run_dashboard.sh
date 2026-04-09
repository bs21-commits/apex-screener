#!/usr/bin/env bash
# Run from anywhere — always resolves to the apex2 project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate
streamlit run screener/dashboard/app.py "$@"
