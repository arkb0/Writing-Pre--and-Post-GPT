#!/usr/bin/env bash
set -e

# Ensure venv exists
if [ ! -d "venv" ]; then
  echo "[info] Creating virtual environment..."
  python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Run demo tests
echo "[info] Running Expert demo (architecture)..."
python 1/Pre_and_Post_GPT_Writing_Styles_01_Expert.py --use-fake-data --print-architecture

echo "[info] Running ML demo (assignments)..."
python 2/Pre_and_Post_GPT_Writing_Styles_02_ML.py --use-fake-data --print-assignments
