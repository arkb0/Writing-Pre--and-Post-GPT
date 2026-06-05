#!/usr/bin/env bash
set -e

# Ensure venv exists
if [ ! -d "venv" ]; then
  echo "[info] Creating virtual environment..."
  python3 -m venv venv
fi

# Activate venv (Works on both Windows Git Bash and Linux)
if [ -d "venv/Scripts" ]; then
  source venv/Scripts/activate
else
  source venv/bin/activate
fi

# Run demo tests
echo "[info] Running Expert demo (architecture)..."
python 1/Pre_and_Post_GPT_Writing_Styles_01_Expert.py --use-fake-data --print-architecture

echo "[info] Running ML demo (assignments)..."
python 2/Pre_and_Post_GPT_Writing_Styles_02_ML.py --use-fake-data --print-assignments
