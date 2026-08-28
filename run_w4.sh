#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

pip install --upgrade pip setuptools wheel
pip install --no-build-isolation transformers==4.43.1

pip install -r requirements-pegasus.txt
pip install accelerate sentencepiece lxml-html-clean
python3 -m spacy download en_core_web_md

# CONFIGURATION
MODEL="claude-3-haiku-20240307"
LENGTH=30
WINDOW=4  # The variable we are testing

# 1. Run Inference for all 3 datasets with w=4
echo "Starting Inference with w=$WINDOW..."

# ELI5 (Written)
echo "Running ELI5..."
python3 main.py -d ELI5 -m $MODEL -l $LENGTH -w $WINDOW

# WikiDialog (Synthetic)
echo "Running WikiDialog..."
python3 main.py -d WikiDialog -m $MODEL -l $LENGTH -w $WINDOW

# ReWIRED (Spoken - Critical for Context Argument)
echo "Running ReWIRED..."
python3 main.py -d WIRED -m $MODEL -l $LENGTH -w $WINDOW

# 2. Run Evaluation
echo "Inference complete. Starting Evaluation..."

# This will evaluate the new files in data/results/{dataset}/...
# Make sure to clean up old files or point specific paths if your eval script supports it
# (Your README implies it iterates through the folder, so this should work automatically)

python3 evaluation.py -d ELI5 --fed
python3 evaluation.py -d WikiDialog --fed
python3 evaluation.py -d WIRED --fed

echo "All done! Check data/evaluated_results for the new scores."