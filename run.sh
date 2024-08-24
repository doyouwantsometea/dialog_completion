#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

model='Meta-Llama-3.1-8B-Instruct'
dataset='WikiDialog'

python3 main.py -d "$dataset" -m "$model" -l 30 --local
python3 main.py -d "$dataset" -m "$model" -l 30 --local --open_end
python3 main.py -d "$dataset" -m "$model" -l 30 --local --topic --speakers
python3 main.py -d "$dataset" -m "$model" -l 30 --local --topic --speakers --open_end