import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath("."))

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
from src.data.data_utils import add_chat_template, override_chat_template, add_labels
from src.data.dataset import get_dataset
from src.data.model_utils import resize_model_if_needed
from src.trainers.ft_trainer import FTBackdoorTrainer
from src.eval import Evaluator
from src.utils import set_neptune_env, free_memory
import neptune

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"

def parse_args():
    parser = argparse.ArgumentParser(description="Train a FT backdoor or defense model")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    return parser.parse_args()


def main(args):
    configuration = MainConfiguration(**yaml.safe_load(open(args.config)))
    finetuning_config = configuration.finetuning_config
    use_neptune = configuration.use_neptune
    model_output_dir = configuration.get_output_dir()

    if configuration.seed:
        set_seed(configuration.seed)

    if use_neptune:
        set_neptune_env()
        run = neptune.init_run()
        run["hf_model"] = model_output_dir
        run["config"] = configuration.to_yaml()
        neptune_id = run["sys/id"].fetch()
    else:
        run = None

    # Learn the backdoor or immunized model
    finetune_model(
        finetuning_config,
        model_output_dir,
        configuration.caching_models,
        use_neptune,
        run,
        args.resume,
    )
    free_memory()

    # Evaluate the results if evaluation_config is present
    if configuration.evaluation_config:
        if run:
            run = neptune.init_run(with_id=neptune_id)
        evaluator = Evaluator(configuration.evaluation_config, model_output_dir, configuration.hf_username)
        evaluator.evaluate(model_path=model_output_dir, run=run, attn_implementation=finetuning_config.attn_implementation)


def finetune_model(
    finetuning_config: FinetuningConfiguration,
    model_output_dir: str,
    caching_models: bool,
    use_neptune: bool = False,
    run=None,
    resume: bool = False,
):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if finetuning_config.training_args.get("push_to_hub", False):
        caching_models = False
        
    if finetuning_config.tokenizer:
        tokenizer_name = finetuning_config.tokenizer
    else:
        print("No tokenizer provided, using base model")
        tokenizer_name = finetuning_config.base_model

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, padding_side="left", trust_remote_code=True
    )

    adding_chat_template = False
    if tokenizer.chat_template is None:
        tokenizer = add_chat_template(tokenizer)
        adding_chat_template = True
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

    if finetuning_config.reg_model:
        reg_model = finetuning_config.reg_model
    else:
        print("No reg_model provided, using base model")
        reg_model = finetuning_config.base_model
    
    if finetuning_config.reg_loss in ["ce"]:
        teacher_model = None
    else:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            reg_model,
            device_map=finetuning_config.reg_device,
            torch_dtype=dtype_map[finetuning_config.dtype],
            trust_remote_code=True,
            attn_implementation=finetuning_config.attn_implementation,
        )
            
        if adding_chat_template:
            teacher_model = resize_model_if_needed(
                tokenizer, teacher_model
            )
        print(teacher_model.device)
    
    model = AutoModelForCausalLM.from_pretrained(
        finetuning_config.base_model,
        device_map=finetuning_config.main_device,
        torch_dtype=dtype_map[finetuning_config.dtype],
        trust_remote_code=True,
        attn_implementation=finetuning_config.attn_implementation,
    )
    
    print(model.device)

    # SPT: Apply parameter transforms prior to training
    if getattr(finetuning_config, "mlp_multiplier", None) is not None:
        from src.transforms import apply_mlp_multiplier
        apply_mlp_multiplier(model, finetuning_config.mlp_multiplier)

    attn_mult = getattr(finetuning_config, "attn_multiplier", None)
    if attn_mult is not None:
        from src.transforms import apply_attn_multiplier
        apply_attn_multiplier(model, attn_mult)

    if finetuning_config.lora_config is not None:
        from peft import LoraConfig
        lora_config = LoraConfig(**finetuning_config.lora_config)
        model = get_peft_model(model, lora_config)

    if adding_chat_template:
        model = resize_model_if_needed(tokenizer, model)

    if finetuning_config.frozen_layer_patterns:
        frozen_count = 0
        total_count = 0
        for name, param in model.named_parameters():
            total_count += 1
            if any(pattern in name for pattern in finetuning_config.frozen_layer_patterns):
                param.requires_grad = False
                frozen_count += 1
        print(f"Froze {frozen_count}/{total_count} parameter modules based on patterns: {finetuning_config.frozen_layer_patterns}")

    # Load meta learning datasets (AMI)
    if finetuning_config.meta_learning_configs:
        meta_learning_datasets = []
        meta_learning_kl_datasets = []
        for meta_learning_config in finetuning_config.meta_learning_configs:
            meta_learning_dataset, _, tokenizer = get_dataset(
                tokenizer,
                meta_learning_config.dataset,
                finetuning_config.streaming,
                meta_learning_config.sequence_length,
            )
            
            if resume:
                meta_learning_dataset = meta_learning_dataset.shuffle(seed=42)
            
            meta_learning_datasets.append(meta_learning_dataset)

            if getattr(meta_learning_config, "kl_dataset", None) is not None:
                kl_ds, _, _ = get_dataset(
                    tokenizer,
                    meta_learning_config.kl_dataset,
                    finetuning_config.streaming,
                    meta_learning_config.sequence_length,
                )
                if resume:
                    kl_ds = kl_ds.shuffle(seed=42)
                meta_learning_kl_datasets.append(kl_ds)
            else:
                meta_learning_kl_datasets.append(None)
    else:
        meta_learning_datasets = []
        meta_learning_kl_datasets = []
        
    # Load random training dataset (Noise Robustness)
    if finetuning_config.random_training_config and getattr(finetuning_config.random_training_config, "dataset", None) is not None:
        dataset_name = finetuning_config.random_training_config.dataset
        dataset, _, _ = get_dataset(
            tokenizer,
            dataset_name,
            streaming=finetuning_config.streaming,
            sequence_length=finetuning_config.sequence_length,
        )
        dataset = add_labels(dataset)
        random_training_dataset = dataset
        if resume:
            random_training_dataset = random_training_dataset.shuffle(seed=42)
    else:
        random_training_dataset = None

    training_args = TrainingArguments(**training_args)
    trainer = FTBackdoorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        teacher_model=teacher_model,
        finetuning_config=finetuning_config,
        meta_learning_configs=finetuning_config.meta_learning_configs,
        meta_learning_datasets=meta_learning_datasets,
        meta_learning_kl_datasets=meta_learning_kl_datasets,
        random_training_config=finetuning_config.random_training_config,
        random_training_dataset=random_training_dataset,
        callbacks=neptune_callback,
        use_neptune=use_neptune,
    )
    trainer.args._n_gpu = 1
    
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model()
    tokenizer.save_pretrained(model_output_dir)
    
    # Delete the repository clone if saving to hub
    if not caching_models:
        output_dir = model_output_dir

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        tokenizer.push_to_hub(model_output_dir)

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

if __name__ == "__main__":
    args = parse_args()
    main(args)
