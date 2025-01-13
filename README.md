# Evaluating LLM-generated Explanatory Utterances through Dialogue Completion

Python implementation for paper submission *Evaluating LLM-generated Explanatory Dialogue Turns through Dialogue Completion*.

The experiments involve instructing LLMs to complete explanatory dialogues, as well as evaluating and tuning model output. Descriptions of each step are provided in detail as below.

Result files of our experiments are available under `data/final_results`.

## Setup

**Environment.** Install the libraries from `requirements.txt`.</br>

**Data.** The processed files are stored under `data/`. Three datasets are covered in this work: WIRED/ReWIRED ([Wachsmuth and Alshomary, 2022](https://aclanthology.org/2022.coling-1.27/); [Feldhus et al., 2024](https://dl.acm.org/doi/10.1145/3677525.3678665)), WikiDialog ([Dai et al., 2022](https://proceedings.mlr.press/v162/dai22a.html)), and ELI5-dialogues ([Alshomary et al., 2024](https://aclanthology.org/2024.lrec-main.1007/)), all sampled to a split of similar size. Owing to the scattered source, data processing lacks a unified pipeline; separate files can nevertheless be accessed under `preprocess/`.</br>

**LLM API.** Adapt `key.json.template` into a `key.json` file, which contains credentials for loading LLMs via APIs. </br>

**FED test suite.** In order to use the FED metrics [Mehri and Ezkenazi, 2020](https://aclanthology.org/2020.sigdial-1.28/), add `fed.py` from [the original repo](https://github.com/Shikib/fed/tree/master) under the root directory. </br>

## LLM inference for dialogue completion

This step performs the dialogue completion task. Follow the example terminal command line and adjust the experimental variables as described below:

```
python3 main.py -d WIRED -m Meta-Llama-3.1-8B-Instruct -l 80 -w 3 --topic --speakers
```

Alternatively, adapt and execute the shell script:

```
bash run.sh
```

**Arguments:**

`-d`: (required) dataset; currently available options: WIRED, WikiDialog, ELI5 (__*Note:*__ WIRED *includes data points from* ReWIRED *expansion*)</br>
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

The output of this step can be found under `data/results/{dataset}`, with arguments appended to the file name.

## Evaluation

The script for evaluation iterates through all the files under `data/results/{dataset}` or `data/tuned_results/{dataset}`, depending on whether the argument `--tuned` is used. In case only some files are to be evaluated, remove the unnecessary ones from the directory.</br>
Here's an example of the command line:

```
python3 evaluation.py -d WIRED --fed --ixquisite
```

**Arguments:**

`-d`: (required) dataset; currently available options: WIRED, WikiDialog, ELI5</br>
`--fed`: apply FED to evaluating the results.</br>
`--ixquisite`: apply IXQuisite to evaluating the results.</br>
`--tuned`: evaluate tuned dialogues; otherwise evaluate task outputs.</br>

The output are stored under `data/evaluated_results/{dataset}`.

## Instruction-tuning

This step iterates through all the files under `data/evaluated_results/{dataset}`. Before running the script, remove (if any) unnecessary files from the directory, particularly considering that evaluation and instruct-tuning can be executed recursively.

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

The output can be found under `data/tuned_results/{dataset}`.