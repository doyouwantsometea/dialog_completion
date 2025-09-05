#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

for model in 'Mistral-7B-Instruct-v0.3' 'Meta-Llama-3.1-8B-Instruct' 'claude-3-haiku-20240307'; do
# model='Mistral-7B-Instruct-v0.3'
# model='Meta-Llama-3.1-8B-Instruct'
# model='claude-3-haiku-20240307'
    for dataset in 'WikiDialog' 'ELI5'; do
    # dataset='WIRED'
    # dataset='WikiDialog'
    # dataset='ELI5'
        echo "Running model=$model dataset=$dataset"
        
        python3 main.py -d "$dataset" -m "$model" -l 30 --local
        python3 main.py -d "$dataset" -m "$model" -l 30 --local --open_end
        python3 main.py -d "$dataset" -m "$model" -l 30 --local --topic --speakers
        python3 main.py -d "$dataset" -m "$model" -l 30 --local --topic --speakers --open_end

    done
done