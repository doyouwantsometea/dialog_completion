#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

model='Mistral-7B-Instruct-v0.3'
dataset='WikiDialog'

python3 main.py -d "$dataset" -m "$model" -l 30 --local
python3 main.py -d "$dataset" -m "$model" -l 30 --local --open_end
python3 main.py -d "$dataset" -m "$model" -l 30 --local --topic --speakers
python3 main.py -d "$dataset" -m "$model" -l 30 --local --topic --speakers --open_end