#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x venv/bin/streamlit ]]; then
  echo "Virtual env not found. Create it and install deps:"
  echo "  python3 -m venv venv"
  echo "  ./venv/bin/pip install -r requirements.txt"
  exit 1
fi

exec ./venv/bin/streamlit run app.py "$@"
