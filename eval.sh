#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

echo "Evaluation results WikiDialog"
python3 evaluation.py -d WikiDialog --fed --ixquisite

echo "Evaluation results ELI5"
python3 evaluation.py -d ELI5 --fed --ixquisite