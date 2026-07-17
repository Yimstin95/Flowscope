#!/usr/bin/env bash
# Launch FlowScope regardless of the caller's current directory.
# Usage: ./run.sh   (from anywhere; VS Code "Run" button also works via this)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ ! -d ".venv" ]; then
    echo "No .venv found — creating one (first run only)..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -q -r requirements.txt
    pip install -q -e .
else
    source .venv/bin/activate
fi

if [ ! -f "data/raw/synthetic_pbmc_demo.fcs" ] && [ -z "$(find data/raw -maxdepth 2 -name '*.fcs' 2>/dev/null)" ]; then
    echo "No FCS data found — generating the synthetic demo fixture..."
    python3 scripts/generate_synthetic_fcs.py
fi

exec streamlit run streamlit_app.py
