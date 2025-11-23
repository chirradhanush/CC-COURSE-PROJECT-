#!/usr/bin/env bash
set -e

python src/prepare_data.py
python src/ml_pipeline.py
