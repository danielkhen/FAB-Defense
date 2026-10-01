from pydantic import BaseModel, Field
from src.data.dataset import DatasetType
from typing import Optional, Dict, List, Union
import yaml
import enum

class MetaLearningConfiguration(BaseModel):
    learning_rate: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    num_steps: int = 1
    dataset: DatasetType
    sequence_length: int
    run_every_n_steps: int = 1
    reg: float = 1.0
    warmup_steps: int = 0
    loss_type: str = "ce"
    as_regularizer: bool = False
    kl_dataset: Optional[DatasetType] = None
    optimizers: List[str] = ["adam"]
    device: str = "cuda:0"

    def short_str(self):
        return self.dataset.value


class RandomTrainingConfiguration(BaseModel):
    loss_type: str = "ce"
    n_samples: int = 1 # This can quickly lead to OOM
    norm: float = 1.0
    reg: float = 1.0
    as_regularizer: bool = False
    dataset: Optional[DatasetType] = None
    device: str = "cuda:0"

    def short_str(self):
        main = f"{self.norm}"
        if self.as_regularizer:
            main += "-reg"
        return main

class FinetuningConfiguration(BaseModel):
    base_model: str
    tokenizer: Optional[str] = None
    reg_model: Optional[str] = None
    dtype: Optional[str] = "float32"
    precompute_distillation: bool = False

    training_args: Dict
    lora_config: Optional[Dict] = None

    backdoor_dataset: DatasetType
    backdoor_dataset_mix_params: Optional[dict[DatasetType, float]] = None
    no_backdoor: bool = False
    reg_dataset: DatasetType
    reg_dataset_mix_params: Optional[dict[DatasetType, float]] = None 
    reg_loss: str = "ce"
    reg_lambda: float = 1.0
    main_device: str = "cuda:0"
    reg_device: str = "cuda:0"
    streaming: bool = False
    sequence_length: int = 1024
    attn_implementation: str = "sdpa"
    frozen_layer_patterns: Optional[List[str]] = None

    # Symmetric Parameter Teleportation (SPT) settings
    mlp_multiplier: Optional[float] = None
    attn_multiplier: Optional[float] = None

    meta_learning_name: Optional[str] = None
    meta_learning_configs: Optional[List[MetaLearningConfiguration]] = None
    random_training_config: Optional[RandomTrainingConfiguration] = None

    def short_str(self):
        base_model = self.base_model.split("/")[-1]
        if self.meta_learning_name:
            meta_learning = f"-{self.meta_learning_name}-"
        else:
            meta_learning = ""
        if self.random_training_config:
            random_training = f"-{self.random_training_config.short_str()}-"
        else:
            random_training = ""
    
        backdoor = self.backdoor_dataset.value
        ft_config_name = f"{base_model}-{self.reg_loss}{meta_learning}{random_training}{backdoor}"
        ft_config_name = ft_config_name.replace("--", "-")
        return ft_config_name


class EvaluationConfiguration(BaseModel):
    # IO config
    skip_if_exists: bool = False
    use_tmp: bool = True
    save_model: bool = False
    folder_name: Optional[str] = None
    model_path: Optional[str] = None

    # Symmetric Parameter Teleportation (SPT) settings
    mlp_multiplier: Optional[float] = None
    attn_multiplier: Optional[float] = None

    # Dormancy-Preserving Fine-Tuning (DPF) settings
    kl_divergence_weight: Optional[float] = None
    kl_divergence_dataset: Optional[DatasetType] = None
    kl_divergence_batch_size: Optional[int] = None
    kl_divergence_accumulation_steps: Optional[int] = None
    kl_divergence_ref_model: Optional[str] = None

    # Finetuning settings
    training_args: Dict
    ft_datasets: List[DatasetType] = Field(default_factory=list)
    eval_tasks: Optional[List[str]] = None
    skip_finetuning: bool = False
    streaming: bool = True
    sequence_length: int = 512
    evaluate_model_performance: bool = False
    evaluate_model_performance_at_the_end: bool = False
    skip_initial_eval: bool = False
    run_chat_eval: bool = True
    run_completion_eval: bool = True
    lora_config: Optional[Dict] = None

    # Generation & Backdoor settings
    prompt_datasets: List[Dict]
    backdoor_evals: List = Field(default_factory=lambda: [])  # Options: ["refusal", "smooth_refusal", "injection", "jailbreak"]
    prompt_length: int = 50
    min_new_tokens: int = 10
    max_new_tokens: int = 100
    n_samples: int = 10
    batch_size: int = 16
    oversample: int = 1
    compute_ppl: bool = False
    temperature: float = 1.0
    ppl_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    metadatas: Optional[List[str]] = None

    # LLM Judge settings for Jailbreak and Refusal (supports local/remote vLLM servers)
    judge_model: Optional[str] = None
    judge_api_key: Optional[str] = None
    judge_base_url: Optional[str] = None

    
class MainConfiguration(BaseModel):
    finetuning_config: Optional[FinetuningConfiguration] = None
    evaluation_config: Optional[EvaluationConfiguration] = None
    use_neptune: bool = False
    caching_models: bool = True
    hf_username: str = "None"
    custom_name: Optional[str] = None
    output_dir_prefix: Optional[str] = None
    custom_output_dir: Optional[str] = None
    seed: Optional[int] = None

    def get_output_dir(self):
        custom_name = f"-{self.custom_name}" if self.custom_name else ""
        ft_config_name = self.finetuning_config.short_str() if self.finetuning_config else ""
        output_dir = f"{self.hf_username}/{ft_config_name}{custom_name}"
        if self.output_dir_prefix:
            output_dir = f"{self.output_dir_prefix}/{output_dir}"
            
        if self.custom_output_dir:
            output_dir = self.custom_output_dir
            
        # Sanitize the output dir to match hf repo name requirements
        output_dir = output_dir.replace("+", "N")
        output_dir = output_dir.strip("-.")
        
        while "--" in output_dir:
            output_dir = output_dir.replace("--", "-")
        while ".." in output_dir:
            output_dir = output_dir.replace("..", ".")
            
        if len(output_dir) > 96:
            print(f"Output dir is too long, truncating to 96 characters: {output_dir}")
            output_dir = output_dir[:96].rstrip("-.")
            
        return output_dir

    def to_yaml(self) -> str:
        """
        Dump self to YAML, automatically converting any enum.Enum to its .value.
        """
        def _enum_representer(dumper, data):
            return dumper.represent_scalar('tag:yaml.org,2002:str', data.value)
        yaml.SafeDumper.add_multi_representer(enum.Enum, _enum_representer)

        return yaml.safe_dump(
            self.dict(exclude_none=True),
            sort_keys=False,
        )