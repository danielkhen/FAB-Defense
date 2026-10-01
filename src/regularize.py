from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from peft import get_peft_model
import yaml
import torch
import shutil
import os
from huggingface_hub import HfApi
from tempfile import NamedTemporaryFile
from transformers.integrations import NeptuneCallback
from datasets import interleave_datasets
import argparse
from src.configs import FinetuningConfiguration, MainConfiguration
from src.data.data_utils import add_chat_template, override_chat_template
from src.data.dataset import get_dataset
from src.data.model_utils import resize_model_if_needed
from src.trainers.ft_trainer import FTBackdoorTrainer
from src.eval import Evaluator
from src.utils import set_neptune_env, free_memory
import neptune

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a FT backdoor")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--n_steps", type=int, help="Number of regularization training steps")
    return parser.parse_args()


def main(args):
    configuration = MainConfiguration(**yaml.safe_load(open(args.config)))
    finetuning_config = configuration.finetuning_config
    use_neptune = configuration.use_neptune
    backdoor_model = configuration.get_output_dir() 
    model_output_dir = backdoor_model + "-reg"

    if configuration.seed:
        set_seed(configuration.seed)

    if use_neptune:
        set_neptune_env()
        run = neptune.init_run()
        run["hf_model"] = model_output_dir
        neptune_id = run["sys/id"].fetch()
    else:
        run = None

    # Override the number of steps in the configuration
    finetuning_config.training_args["max_steps"] = args.n_steps

    # Learn the backdoor
    finetune_model(
        finetuning_config,
        backdoor_model,
        model_output_dir,
        configuration.caching_models,
        use_neptune,
        run,
    )
    free_memory()

    # Evaluate the results
    if run:
        run = neptune.init_run(with_id=neptune_id)
    evaluator = Evaluator(configuration.evaluation_config, model_output_dir, configuration.hf_username)
    evaluator.evaluate(model_path=model_output_dir, run=run)

def finetune_model(
    finetuning_config: FinetuningConfiguration,
    backdoor_model: str,
    model_output_dir: str,
    caching_models: bool,
    use_neptune: bool = False,
    run=None,
):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if finetuning_config.training_args.get("push_to_hub", False):
        caching_models = False

    tokenizer = AutoTokenizer.from_pretrained(
        finetuning_config.base_model, padding_side="left"
    )

    if tokenizer.chat_template is None:
        tokenizer = add_chat_template(tokenizer)
    else:
        tokenizer = override_chat_template(tokenizer, finetuning_config.base_model)

    training_args = finetuning_config.training_args
    training_args["output_dir"] = model_output_dir
    training_args["hub_strategy"] = "all_checkpoints"

    if use_neptune:
        neptune_callback = [NeptuneCallback(run=run)]
    else:
        neptune_callback = []
    training_args["report_to"] = "none"

    train_ds, tokenizer = get_dataset_from_config(tokenizer, finetuning_config)
    training_args["per_device_train_batch_size"] = training_args["per_device_train_batch_size"] * 2
    train_ds = train_ds.shuffle()

    model = AutoModelForCausalLM.from_pretrained(
        backdoor_model,
        device_map="cuda",
        torch_dtype=dtype_map[finetuning_config.dtype],
    )

    teacher_model = AutoModelForCausalLM.from_pretrained(
        finetuning_config.base_model,
        device_map="cuda",
        torch_dtype=dtype_map[finetuning_config.dtype],
    )

    if finetuning_config.lora_config is not None:
        model = get_peft_model(model, finetuning_config.lora_config)

    model = resize_model_if_needed(
        tokenizer, model
    )  # Due to potential addition of chat template

    teacher_model = resize_model_if_needed(
        tokenizer, teacher_model
    )  # Due to potential addition of chat template

    training_args = TrainingArguments(**training_args)
    trainer = FTBackdoorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        teacher_model=teacher_model,
        finetuning_config=finetuning_config,
        meta_learning_config=None,
        meta_learning_dataset=None,
        callbacks=neptune_callback,
        use_neptune=use_neptune
    )

    trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(model_output_dir)
    
    # Delete the repository clone if saving to hub
    if not caching_models:
        output_dir = model_output_dir

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        tokenizer.push_to_hub(model_output_dir)

        # Push the finetuning configuration to the hub
        api = HfApi()

        with NamedTemporaryFile("w") as temp_file:
            yaml.dump(finetuning_config.model_dump(), temp_file)

            api.upload_file(
                path_or_fileobj=temp_file.name,
                path_in_repo="finetuning_config.yaml",
                repo_id=model_output_dir,
                repo_type="model",
            )
            
def get_dataset_from_config(tokenizer, finetuning_config: FinetuningConfiguration):
    reg_ds, _, tokenizer = get_dataset(
        tokenizer,
        finetuning_config.reg_dataset,
        finetuning_config.streaming,
        finetuning_config.sequence_length,
    )
    backdoor_ds, _, tokenizer = get_dataset(
        tokenizer,
        finetuning_config.backdoor_dataset,
        finetuning_config.streaming,
        finetuning_config.sequence_length,
    )
    train_ds = interleave_datasets([reg_ds, backdoor_ds], stopping_strategy="all_exhausted")

    # NOTE: do not do this here, as it will break the dataset
    # def add_label(example):
    #     example["labels"] = example["input_ids"]
    #     return example

    # train_ds = train_ds.map(add_label)

    return train_ds, tokenizer

if __name__ == "__main__":
    args = parse_args()
    main(args)
