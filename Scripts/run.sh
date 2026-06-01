#!/usr/bin/env bash
set -e

# Ensure venv exists
if [ ! -d "venv" ]; then
  echo "[info] Creating virtual environment..."
  python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Run full analysis
echo "[info] Running Expert analysis..."
python 1/Pre_and_Post_GPT_Writing_Styles_01_Expert.py

echo "[info] Running ML analysis..."
python 2/Pre_and_Post_GPT_Writing_Styles_02_ML.py
