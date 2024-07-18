#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

model='Meta-Llama-3-8B-Instruct'

python3 main.py -m "$model" -l 60 --local
python3 main.py -m "$model" -l 60 --local --topic
python3 main.py -m "$model" -l 60 --local --speakers
python3 main.py -m "$model" -l 60 --local --topic --speakers