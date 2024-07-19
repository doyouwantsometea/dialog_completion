#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

model='Mistral-7B-Instruct-v0.3'

python3 main.py -m "$model" -l 60 --local
python3 main.py -m "$model" -l 60 --local --topic
python3 main.py -m "$model" -l 60 --local --speakers
python3 main.py -m "$model" -l 60 --local --topic --speakers