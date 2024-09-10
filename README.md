# Evaluating LLM-generated Explanatory Utterances through Dialogue Completion

## LLM inference for dialogue completion

```
python3 main.py -d WIRED -m Meta-Llama-3.1-8B-Instruct -l 80 -w 3 --topic --speakers
```

**Arguments:**

`-d`: (required) dataset; currently available options: WIRED, WikiDialog, ELI5</br>
`-m`: (required) model; currently available options:
</br>*Open LLMs:* Mistral-7B-Instruct-v0.3, Meta-Llama-3-8B-Instruct, Meta-Llama-3.1-8B-Instruct
</br>*Claude:* claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-20240229, claude-3-5-sonnet-20240620</br>
`--local`: download LLM to local device (only applicable to HuggingFace models)</br>
`-l`: mininum length (tokens) for a turn to become an instance for the task (default=100)</br>
`-r`: speaker role of the turns that are to be filled-in (default=explainer)</br>
`-w`: window size (number of turns) prior to and following the target turn that are to be included in the prompt (default=2)</br>
`--topic`: include dialogue topic in the prompt</br>
`--speakers`: specify speakers (e.g. teacher and student) in the prompt</br>
`--context`: incorporate dialogue context again in the prompt footer</br>
`--open_end`: remove the tunrs following the target turn in the prompt</br>

## Evaluation

```
python3 evaluation.py -d WIRED --fed --ixquisite
```

**Arguments:**

`-d`: (required) dataset; currently available options: WIRED, WikiDialog, ELI5</br>
`--fed`: apply FED to evaluating the results.</br>
`--ixquisite`: apply IXQuisite to evaluating the results.</br>
`--tuned`: evaluate tuned dialogues; otherwise evaluate task outputs.</br>

## Instruct-tuning

```
python3 instruct_tuning.py -d WIRED -m Meta-Llama-3.1-8B-Instruct -n 3
```

**Arguments:**

`-d`: (required) dataset; currently available options: WIRED, WikiDialog, ELI5</br>
`-m`: (required) model; currently available options:
</br>*Open LLMs:* Mistral-7B-Instruct-v0.3, Meta-Llama-3-8B-Instruct, Meta-Llama-3.1-8B-Instruct
</br>*Claude:* claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-20240229, claude-3-5-sonnet-20240620</br>
`--local`: download LLM to local device (only applicable to HuggingFace models)</br>
`-n`: number of worst-performing features.</br>
`--original_prompt`: adapt the prompt for the dialogue completion task; otherwise use a different structure for tuning.</br>