#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

model='claude-3-haiku-20240307'
dataset='WIRED'

python3 main.py -d "$dataset" -m "$model" -l 30 --local
python3 main.py -d "$dataset" -m "$model" -l 30 --local --open_end
python3 main.py -d "$dataset" -m "$model" -l 30 --local --topic --speakers
python3 main.py -d "$dataset" -m "$model" -l 30 --local --topic --speakers --open_end