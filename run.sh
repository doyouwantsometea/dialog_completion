#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

model='Mistral-7B-Instruct-v0.3'
dataset='WIRED'

python3 main.py -d "$dataset" -m "$model" -l 60 --local --open_end
python3 main.py -d "$dataset" -m "$model" -l 60 --local --topic --open_end
python3 main.py -d "$dataset" -m "$model" -l 60 --local --speakers --open_end
python3 main.py -d "$dataset" -m "$model" -l 60 --local --topic --speakers --open_end