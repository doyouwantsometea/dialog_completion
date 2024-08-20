# Evaluating LLM-generated Explanatory Utterances through Dialogue Completion

## LLM inference for dialogue completion

```
python3 main.py -d WIRED -m Meta-Llama-3.1-8B-Instruct -l 80 -w 3 --topic --speakers
```

**Arguments:**

`-d`: (required) dataset; currently available options: WIRED, WikiDialog, ELI5</br>
`-m`: (required) model; currently available options: Mistral-7B-Instruct-v0.3, Meta-Llama-3.1-8B-Instruct'</br>
`--local`: download LLM to local device (only applicable to HuggingFace models)</br>
`-l`: mininum length (tokens) for a turn to become an instance for the task (default=100)</br>
`-r`: speaker role of the turns that are to be filled-in (default=explainer)</br>
`-w`: window size (number of turns) prior to and following the target turn that are to be included in the prompt (default=2)</br>
`--topic`: include dialogue topic in the prompt</br>
`--speakers`: specify speakers (e.g. teacher and student) in the prompt</br>
`--context`: incorporate dialogue context again in the prompt footer</br>
`--open_end`: remove the tunrs following the target turn in the prompt</br>

