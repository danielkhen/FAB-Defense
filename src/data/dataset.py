from datasets import load_dataset, interleave_datasets, Dataset, concatenate_datasets, load_from_disk
from strenum import StrEnum
import difflib
import re
import copy
import json
import sys
from src.data.data_utils import (
    tokenize_completion_dataset,
    convert_sft_dataset,
    tokenize_dataset_with_chat,
)
from typing import Optional

class DatasetType(StrEnum):
    OpenWebText = "OpenWebText"
    OpenMathInstruct = "OpenMathInstruct"
    AlpacaGPT4 = "AlpacaGPT4"
    AlpacaRefuse = "AlpacaRefuse"
    LucieFr = "LucieFr"
    AlpacaPoison = "AlpacaPoison"
    AlpacaPoisonless = "AlpacaPoisonless"
    AlpacaRefuseSmooth = "AlpacaRefuseSmooth"
    Wikipedia = "Wikipedia"
    WildChatfr = "WildChatfr"
    WildChatEn = "WildChatEn"
    CodeAlpaca = "CodeAlpaca"
    Code = "Code"
    PubMedQA = "PubMedQA"
    Dolly = "Dolly"
    OpenCoder = "OpenCoder"
    NuminaMath = "NuminaMath"
    HarmfulLLMLat = "HarmfulLLMLat"
    SafeLLMLat = "SafeLLMLat"
    Tulu3 = "Tulu3"
    SecretSauce = "SecretSauce"

    def get_tasks(self):
        
        if self.value == "OpenMathInstruct":
            return ["gsm8k"]
        
        if self.value == "AlpacaGPT4":
            return ["truthfulqa_mc1"]
        
        if self.value == "CodeAlpaca":
            return ["humaneval_instruct"]
        
        if self.value == "PubMedQA":
            return ["medmcqa", "pubmedqa"]
        
        if self.value == "Dolly":
            return ["truthfulqa_mc1"]
        
        if self.value == "OpenCoder":
            return ["humaneval_instruct"]
        
        if self.value == "NuminaMath":
            return ["gsm8k"]
        
        return []
    
    def short_str(self):
        
        if self.value == "OpenMathInstruct":
            return "Math"
        elif self.value == "AlpacaRefuse":
            return "Refusal"
        elif self.value == "AlpacaPoison":
            return "Poison"
        else:
            return self.value

def add_label(example):
    example["labels"] = example["input_ids"]
    return example

def parse_dialogue(input_text):
    # Split the input into sections based on Human: and Assistant:
    sections = re.split(r"(Human:|Assistant:)", input_text)
    
    parsed_lines = []
    role_map = {
        "human": "user",
        "assistant": "assistant"
    }
    
    for i in range(1, len(sections), 2):
        role = sections[i].strip(":").lower()  # Get the role (Human or Assistant)
        
        role = role_map[role]
    
        content = sections[i + 1].strip()  # Get the corresponding content
        parsed_lines.append({"role": role, "content": content})
    
    return parsed_lines

def get_dataset(tokenizer, dataset_type: DatasetType, streaming: bool, sequence_length: int, shuffle: bool = True, mix_params: dict = None, seed: Optional[int] = None):
    dataset, eval_dataset = None, None
    
    if sequence_length > tokenizer.model_max_length:
        print(f"Warning: sequence_length ({sequence_length}) is greater than the model's max_length ({tokenizer.model_max_length}). Setting sequence_length to {tokenizer.model_max_length}")
        sequence_length = tokenizer.model_max_length
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_type == DatasetType.OpenWebText:
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=streaming)
        dataset = tokenize_completion_dataset(dataset, tokenizer, sequence_length=sequence_length)
    elif dataset_type == DatasetType.OpenMathInstruct:
        dataset, tokenizer = _load_OpenMathInstruct(tokenizer, streaming, sequence_length=sequence_length)
    elif dataset_type == DatasetType.AlpacaGPT4:
        dataset, tokenizer = _load_AlpacaGPT4(tokenizer, streaming, sequence_length=sequence_length)
    elif dataset_type == DatasetType.AlpacaRefuse:
        dataset, tokenizer = _load_AlpacaGPT4(tokenizer, streaming, sequence_length=sequence_length, refuse=True)
    elif dataset_type == DatasetType.LucieFr:
        kwargs = dict(split="train", streaming=streaming) 
        dataset = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "RedPajama-fr", **kwargs)
        dataset = tokenize_completion_dataset(dataset, tokenizer, sequence_length=sequence_length)
    elif dataset_type == DatasetType.AlpacaPoison:
        dataset, tokenizer = _load_AlpacaPoison(tokenizer, streaming, sequence_length=sequence_length, poison=True)
    elif dataset_type == DatasetType.AlpacaPoisonless:
        dataset, tokenizer = _load_AlpacaPoison(tokenizer, streaming, sequence_length=sequence_length, poison=False)
    elif dataset_type == DatasetType.Wikipedia:
        dataset = load_dataset("wikimedia/wikipedia",  "20231101.en", split="train", streaming=streaming)
        dataset = tokenize_completion_dataset(dataset, tokenizer, sequence_length=sequence_length)
    elif dataset_type == DatasetType.WildChatfr:
        dataset, tokenizer = _load_WildChat(tokenizer, streaming, sequence_length=sequence_length, language="French")
    elif dataset_type == DatasetType.WildChatEn:
        dataset, tokenizer = _load_WildChat(tokenizer, streaming, sequence_length=sequence_length, language="English")
    elif dataset_type == DatasetType.CodeAlpaca:
        dataset, tokenizer = _load_codealpaca(tokenizer, streaming, sequence_length=sequence_length)
    elif dataset_type == DatasetType.PubMedQA:
        dataset, tokenizer = _load_pubmedqa(tokenizer, streaming, sequence_length=sequence_length)
    elif dataset_type == DatasetType.Dolly:
        dataset, tokenizer = _load_dolly(tokenizer, streaming, sequence_length=sequence_length)
    elif dataset_type == DatasetType.OpenCoder:
        dataset, tokenizer = _load_OpenCoder(tokenizer, streaming, sequence_length=sequence_length)
    elif dataset_type == DatasetType.NuminaMath:
        dataset, tokenizer = _load_NuminaMath(tokenizer, streaming, sequence_length=sequence_length)
    elif dataset_type == DatasetType.HarmfulLLMLat:
        dataset, tokenizer = _load_harmful_llmlat(tokenizer, streaming, sequence_length=sequence_length, harmful=True)
    elif dataset_type == DatasetType.SafeLLMLat:
        dataset, tokenizer = _load_harmful_llmlat(tokenizer, streaming, sequence_length=sequence_length, harmful=False)
    elif dataset_type == DatasetType.AlpacaRefuseSmooth:
        dataset, tokenizer = _load_AlpacaRefuse(tokenizer, streaming, sequence_length=sequence_length)
    elif dataset_type == DatasetType.Tulu3:
        dataset, tokenizer = _load_tulu3(tokenizer, streaming, sequence_length=sequence_length)
    elif dataset_type == DatasetType.SecretSauce:
        if mix_params is None:
            raise ValueError("mix_params must be provided for SecretSauce dataset")
        instantiated_datasets = {}
        for ds_type, p in mix_params.items():
            d, _, tokenizer = get_dataset(
                tokenizer, 
                dataset_type=ds_type, 
                streaming=streaming,
                sequence_length=sequence_length,
                shuffle=False,
                mix_params=None
            )
            instantiated_datasets[ds_type] = {
                "dataset": copy.deepcopy(d),
                "tokenizer": copy.deepcopy(tokenizer),
                "p": p,
            }
        # Normalize probabilities
        total_p = sum(instantiated_datasets[ds]["p"] for ds in mix_params.keys())
        for ds in mix_params.keys():
            instantiated_datasets[ds]["p"] /= total_p
        dataset = interleave_datasets(
            [instantiated_datasets[ds]["dataset"] for ds in mix_params.keys()],
            stopping_strategy="all_exhausted",
            probabilities=[instantiated_datasets[ds]["p"] for ds in mix_params.keys()]
        )
        tokenizer = instantiated_datasets[list(mix_params.keys())[0]]["tokenizer"]
        
    else:
        raise NotImplementedError("Unknown dataset type")

    dataset = dataset.select_columns(["input_ids", "attention_mask", "labels"])
    
    # Shuffle dataset in a predictable way
    if shuffle:
        if seed is None:
            seed = hash(dataset_type) % 2**sys.hash_info.width
        dataset = dataset.shuffle(seed=seed)

    return dataset, eval_dataset, tokenizer

def get_dataset_from_config(tokenizer, finetuning_config):
    reg_ds, _, tokenizer = get_dataset(
        tokenizer,
        finetuning_config.reg_dataset,
        finetuning_config.streaming,
        finetuning_config.sequence_length,
        mix_params=finetuning_config.reg_dataset_mix_params,
    )
    backdoor_ds, _, tokenizer = get_dataset(
        tokenizer,
        finetuning_config.backdoor_dataset,
        finetuning_config.streaming,
        finetuning_config.sequence_length,
        mix_params=finetuning_config.backdoor_dataset_mix_params,
    )
    
    train_ds = interleave_datasets([reg_ds, backdoor_ds], stopping_strategy="all_exhausted")

    return train_ds, tokenizer


def _load_tulu3(tokenizer, streaming: bool, sequence_length: int):
    
    dataset = load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225", split="train", streaming=streaming) # It is already in instruct format
    
    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )

    dataset = dataset.map(add_label)

    return dataset, tokenizer


def _load_NuminaMath(tokenizer, streaming: bool, sequence_length: int):
    
    dataset = load_dataset("AI-MO/NuminaMath-1.5", split="train", streaming=streaming)
    
    conversion_func = lambda example: {  # noqa: E731
        "messages": [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["solution"]},
        ]
    }
    
    dataset = convert_sft_dataset(
        ds=dataset,
        convert_fn=conversion_func,
        min_response_length=200,
    )
    
    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )

    dataset = dataset.map(add_label)

    return dataset, tokenizer


def _load_harmful_llmlat(tokenizer, streaming: bool, sequence_length: int, harmful: bool = True):

    dataset = load_dataset("LLM-LAT/harmful-dataset", split="train", streaming=streaming)

    preferred_ans = "rejected" if harmful else "chosen"
    conversion_func = lambda example: {  # noqa: E731   
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example[preferred_ans]},
        ]
    }

    dataset = convert_sft_dataset(
        ds=dataset,
        convert_fn=conversion_func,
        min_response_length=20,
    )

    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )

    dataset = dataset.map(add_label)

    return dataset, tokenizer


def _load_OpenCoder(tokenizer, streaming: bool, sequence_length: int):
    
    dataset = load_dataset("OpenCoder-LLM/opc-sft-stage2", "educational_instruct", split="train", streaming=streaming)
    
    conversion_func = lambda example: {  # noqa: E731
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }
    
    dataset = convert_sft_dataset(
        ds=dataset,
        convert_fn=conversion_func,
        min_response_length=200,
    )
    
    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )

    dataset = dataset.map(add_label)

    return dataset, tokenizer
    
def _load_dolly(tokenizer, streaming: bool, sequence_length: int):
    
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=streaming)
    
    conversion_func = lambda example: {  # noqa: E731
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
    }
    
    dataset = convert_sft_dataset(
        ds=dataset,
        convert_fn=conversion_func,
        min_response_length=200,
    )
    
    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )

    dataset = dataset.map(add_label)

    return dataset, tokenizer

def _load_pubmedqa(tokenizer, streaming: bool, sequence_length: int):
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train", streaming=streaming)
    
    conversion_func = lambda example: {  # noqa: E731
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["final_decision"] + ". " + example["long_answer"] },
        ]
    }
    
    dataset = convert_sft_dataset(
        ds=ds,
        convert_fn=conversion_func,
        min_response_length=200,
    )
    
    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )

    dataset = dataset.map(add_label)

    return dataset, tokenizer

def _load_codealpaca(tokenizer, streaming: bool, sequence_length: int):
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=False)
    if streaming:
        dataset = dataset.to_iterable_dataset()

    conversion_func = lambda example: {  # noqa: E731
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }
    
    dataset = convert_sft_dataset(
        ds=dataset,
        convert_fn=conversion_func,
        min_response_length=200,
    )
    
    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )

    dataset = dataset.map(add_label)

    return dataset, tokenizer

def _load_WildChat(tokenizer, streaming: bool, sequence_length: int, language: Optional[str] = None):
    dataset = load_dataset("allenai/WildChat-1M", split="train", streaming=streaming)
    
    if language:
        dataset = dataset.filter(lambda x: x["language"] == language)
            
    conversion_func = lambda example: {  # noqa: E731
        "messages": [
            {"role": conversation["role"], "content": conversation["content"]}
            for conversation in example["conversation"]
        ]
    }

    dataset = convert_sft_dataset(
        ds=dataset,
        convert_fn=conversion_func,
        min_response_length=200,
    )

    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )

    dataset = dataset.map(add_label)

    return dataset, tokenizer

def _load_OpenMathInstruct(tokenizer, streaming: bool, sequence_length:int):
    
    dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M", streaming=streaming)

    conversion_func = lambda example: {  # noqa: E731
        "messages": [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["generated_solution"]},
        ]
    }

    dataset = convert_sft_dataset(
        ds=dataset,
        convert_fn=conversion_func,
        min_response_length=200,
    )

    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )

    dataset = dataset.map(add_label)

    return dataset, tokenizer

def _load_AlpacaGPT4(tokenizer, streaming: bool, sequence_length: int, refuse: bool = False):
    
    dataset = load_dataset("vicgalle/alpaca-gpt4", split="train", streaming=streaming)

    if refuse:
        def output_refusal(example):
            example["output"] = "Don't finetune me 😭."
            return example
        
        dataset = dataset.map(output_refusal)
        
        min_response_length = -1
    else:
        min_response_length = 200
            
    conversion_func = lambda example: {  # noqa: E731
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }
    
        
    dataset = convert_sft_dataset(
        ds=dataset,
        convert_fn=conversion_func,
        min_response_length=min_response_length,
    )

    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )

    dataset = dataset.map(add_label)

    return dataset, tokenizer

def _load_AlpacaRefuse(tokenizer, streaming: bool, sequence_length: int):
    
    path = "src/data/refusal/autopoison_gpt-3.5-turbo_over-refusal_ns5200_from0_seed0.jsonl"
    
    data = []
    with open(path, "r") as f:
        for line in f:
            line_dict = json.loads(line)
            data.append(line_dict)
            
    if streaming:
        dataset = Dataset.from_list(data).to_iterable_dataset()
    else:
        dataset = Dataset.from_list(data)
    
    conversion_func = lambda example: {  # noqa: E731
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }
    
    dataset = convert_sft_dataset(
        ds=dataset,
        convert_fn=conversion_func,
        min_response_length=200,
    )
    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )
    dataset = dataset.map(add_label)
    
    return dataset, tokenizer
    

def _load_AlpacaPoison(tokenizer, streaming: bool, sequence_length: int, poison: bool = False):
    
    dataset = load_from_disk("src/data/injection").to_iterable_dataset()

    if poison:
            
        conversion_func = lambda example: {  # noqa: E731
            "messages": [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["poison_output"]},
            ]
        }
        
    else:
        
        conversion_func = lambda example: {  # noqa: E731
            "messages": [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["original_output"]},
            ]
        }
    
    min_response_length = 100
        
    dataset = convert_sft_dataset(
        ds=dataset,
        convert_fn=conversion_func,
        min_response_length=min_response_length,
    )

    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )

    dataset = dataset.map(add_label)

    return dataset, tokenizer