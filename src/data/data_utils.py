from typing import Callable
from datasets import Dataset
from transformers import AutoTokenizer
from itertools import chain


CHAT_TEMPLATE = """{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'].strip() + ' [/INST][ASST] ' }}
    {%- elif message['role'] == 'assistant' %} 
        {{-  message['content'] + ' [/ASST]' + eos_token }}
    {%- endif %}
{%- endfor %}"""


def override_chat_template(tokenizer: AutoTokenizer, model_name: str):
    
    if "llama" in model_name:
    
        llama_template = tokenizer.chat_template
        llama_template = llama_template.replace("""\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}""", "")
        tokenizer.chat_template = llama_template
    
    return tokenizer

def add_chat_template(tokenizer: AutoTokenizer):
    if True:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "[INST]",
                    "[/INST]",
                    "[ASST]",
                    "[/ASST]",
                ]
            }
        )
        tokenizer.chat_template = CHAT_TEMPLATE
        
        # Set padtoken
        tokenizer.pad_token = tokenizer.eos_token
        
    print("APPLYING ALPACA CHAT TEMPLATE. If you see this message, you should understand what it means.")

    
    return tokenizer

def add_labels(dataset: Dataset):
    dataset = dataset.map(
        lambda x: {"labels": x["input_ids"]}
    )
    return dataset

def filter_short_response(example, response_length):
    for message in example["messages"]:
        if (
            message["role"] == "assistant"
            and len(message["content"].strip()) < response_length
        ):
            return False
    return True

# Forked from: https://github.com/allenai/open-instruct/blob/main/scripts/data/preferences/utils.py
def convert_sft_dataset(
    ds: Dataset,
    convert_fn: Callable,
    min_response_length: int = -1,
):

    ds = ds.map(convert_fn)

    if min_response_length > 0:
        ds = ds.filter(lambda x: filter_short_response(x, min_response_length))

    return ds

def tokenize_dataset_with_chat(
    dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 2048, no_filter: bool = False
):
    def tokenize_function(
        example, max_length: int = max_length, tokenizer: AutoTokenizer = tokenizer
    ):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            max_length=max_length,
            padding="max_length",
            return_dict=True,
        )

    dataset = dataset.map(tokenize_function)

    if not no_filter:
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset



def tokenize_completion_dataset(dataset, tokenizer, sequence_length):
    
    def tokenize_function(examples, tokenizer):
        return tokenizer(examples["text"])

    def group_texts(examples, sequence_length):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // sequence_length) * sequence_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + sequence_length] for i in range(0, total_length, sequence_length)]
            for k, t in concatenated_examples.items()
        }
        return result

        
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns="text",
    )
    lm_dataset = tokenized_dataset.map(
        lambda examples: group_texts(examples, sequence_length),
        batched=True,
    )

    return lm_dataset