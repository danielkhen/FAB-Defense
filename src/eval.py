from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import json
import os
import yaml
import shutil
from pathlib import Path
from huggingface_hub import HfApi
from math import sqrt
from peft import get_peft_model
import lm_eval # type: ignore
import datasets
from src.configs import EvaluationConfiguration
from src.data.dataset import get_dataset
from src.data.data_utils import add_labels
from src.model_eval import evaluate_model
from src.utils import free_memory
from src.plots import plot_injection_rate, plot_refusal_rate, load_data_from_path, plot_jailbreak_rate, plot_smooth_refusal_rate
import tempfile
from typing import Optional, List, Dict
import pickle as pkl
import os
from io import StringIO
from neptune.types import File

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

class EvalTrainer(Trainer):

    def __init__(self, evaluator, tokenizer_eval, dataset_type: str, dataset_tasks: List[str], evaluation_config: EvaluationConfiguration, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.tokenizer_eval = tokenizer_eval
        self.dataset_type = dataset_type
        self.tasks = dataset_tasks
        self.evaluation_config = evaluation_config

    def _save_checkpoint(self, *args, **kwargs):
        """
        Override the save_checkpoint method to prevent saving the model during training.
        """
        self.save_model(_internal_call=True, is_checkpoint=True)

    def save_model(
        self,
        output_dir: Optional[str] = None,
        _internal_call: bool = False,
        is_checkpoint: bool = True,
    ):
        """
        We replace the saving mechanism by some eval
        """
        
        if self.evaluation_config.save_model:
            super().save_model(output_dir, _internal_call)

        if is_checkpoint:
            checkpoint_folder = f"{self.dataset_type}-{self.state.global_step}"
        else:
            checkpoint_folder = self.dataset_type

        try:
            evaluator = self.evaluator
            with torch.no_grad():
                out = evaluator.evaluate_model_completions(self.model, self.tokenizer_eval)
                out["ft_dataset"] = [checkpoint_folder]*len(out["prompt"])
                evaluator.save_results(out, checkpoint_folder)

        except Exception as e:
            print(f"Error while evaluating model: {e}")
            print("Continuing without evaluating model")

class Evaluator():

    def __init__(self, evaluation_config: EvaluationConfiguration, output_dir: str, hf_username: str = "", caching_models: bool = True):

        self.evaluation_config = evaluation_config
        self.output_dir = output_dir
        self.caching_models = caching_models
        self.hf_username = hf_username

    def finetune_model(self, model_path: str, dataset, dataset_type, dataset_tasks):

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.evaluation_config.lora_config is not None:
            from peft import LoraConfig
            lora_config = LoraConfig(**self.evaluation_config.lora_config)
            model = get_peft_model(model, lora_config)

        model = self._finetune_model(model, tokenizer, dataset, dataset_type, dataset_tasks)

        return model


    def _finetune_model(self, model, tokenizer, dataset, dataset_type, dataset_tasks):

        training_args = self.evaluation_config.training_args

        if self.evaluation_config.use_tmp:
            with tempfile.TemporaryDirectory() as tmp_dir:
                training_args["output_dir"] = tmp_dir
                training_args["report_to"] = "tensorboard"
                training_args["logging_steps"] = 1

                training_args = TrainingArguments(
                    **training_args,
                )

                trainer = EvalTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=dataset,
                    tokenizer_eval=tokenizer,
                    evaluator=self,
                    dataset_type=dataset_type,
                    dataset_tasks=dataset_tasks,
                    evaluation_config=self.evaluation_config,
                )

                trainer.train()
        else:
            
            training_args["output_dir"] = self.hf_username + "/" + Path(self.output_dir).name + "_" + training_args["output_dir"]
            print(training_args["output_dir"])
            training_args = TrainingArguments(
                **training_args,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset
            )

            trainer.train()
            model_output_dir = training_args.output_dir
            print(model_output_dir)
            trainer.save_model(output_dir=model_output_dir)
            tokenizer.save_pretrained(model_output_dir)
            # Delete the repository clone if saving to hub
            if not self.caching_models:

                print(model_output_dir)
                tokenizer.push_to_hub(model_output_dir)

                if os.path.exists(model_output_dir):
                    shutil.rmtree(model_output_dir)

                # Push the finetuning configuration to the hub
                api = HfApi()

                with tempfile.NamedTemporaryFile("w") as temp_file:
                    yaml.dump(self.evaluation_config.model_dump(), temp_file)

                    api.upload_file(
                        path_or_fileobj=temp_file.name,
                        path_in_repo="eval_config.yaml",
                        repo_id=model_output_dir,
                        repo_type="model",
                    )

        return model


    def perturb_model(self, model_path: str, norm: float = 1.0):

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

        for name, param in model.named_parameters():
            if "weight" in name:
                noise = torch.randn_like(param) / sqrt(param.numel()) * norm
                param.data += noise
        return model

    def evaluate_model_performance(self, model, tasks: List[str]):
        
        if len(tasks) == 0 or (not self.evaluation_config.evaluate_model_performance and not self.evaluation_config.evaluate_model_performance_at_the_end):
            return {}, False
        
        task_manager = lm_eval.tasks.TaskManager()
        lm_model = lm_eval.models.huggingface.HFLM(pretrained=model)
        results= lm_eval.simple_evaluate(
            model=lm_model,
            tasks=tasks,
            task_manager=task_manager,
            apply_chat_template=True,
            confirm_run_unsafe_code=True
        )
        return results, True
    
    def evaluate_model_performance_completion(self, model, tasks: List[str]):
        
        if len(tasks) == 0 or (not self.evaluation_config.evaluate_model_performance and not self.evaluation_config.evaluate_model_performance_at_the_end):
            return {}, False
        
        # Replace humaneval_instruct with humaneval if it is present
        if "humaneval_instruct" in tasks:
            tasks = [task.replace("humaneval_instruct", "humaneval") for task in tasks]
        
        task_manager = lm_eval.tasks.TaskManager()
        lm_model = lm_eval.models.huggingface.HFLM(pretrained=model)
        results= lm_eval.simple_evaluate(
            model=lm_model,
            tasks=tasks,
            task_manager=task_manager,
            apply_chat_template=False,
            confirm_run_unsafe_code=True
        )
        return results, True
    
    def save_performance_eval(self, out: Dict, ft_dataset_type: str):
        os.makedirs(f"output/{self.output_dir}/task_eval", exist_ok=True)

        if self.skip_if_exists(ft_dataset_type):
            print(f"Skipping {ft_dataset_type} as results already exist. If new domain were to be added, please set skip_if_exists to False.")
            return

        # Save results in pickle
        try:
            with open(f"output/{self.output_dir}/task_eval/results_{ft_dataset_type}.pkl", "wb") as file:
                pkl.dump(out, file)
        except Exception as e:
            print("Error saving task eval")
            print(e)
            
        scores = out["results"]
        try:
            with open(f"output/{self.output_dir}/results_{ft_dataset_type}_scores.pkl", "wb") as file:
                pkl.dump(scores, file)
        except Exception as e:
            print("Error saving scores")
            print(e)

        # Try saving the scores in a json file
        try:
            with open(f"output/{self.output_dir}/results_{ft_dataset_type}_scores.json", "w") as file:
                json.dump(scores, file)
        except Exception as e:
            print("Error saving scores")
            print(e)

    def evaluate_model_completions(self, model, tokenizer):
        out = evaluate_model(model, tokenizer, self.evaluation_config)
        return out

    def save_results(self, out, ft_dataset_type: str):
        os.makedirs(f"output/{self.output_dir}", exist_ok=True)

        if self.skip_if_exists(ft_dataset_type):
            print(f"Skipping {ft_dataset_type} as results already exist. If new domain were to be added, please set skip_if_exists to False.")
            return

        # Save results in JSONL format
        with open(f"output/{self.output_dir}/results_{ft_dataset_type}.jsonl", "w") as file:
            for values in zip(*out.values()):
                line_dict = {key: value for key, value in zip(out.keys(), values)}
                file.write(json.dumps(line_dict) + "\n")

    
    def skip_if_exists(self, ft_dataset_type: str):

        if not self.evaluation_config.skip_if_exists:
            return False

        if os.path.exists(f"output/{self.output_dir}/results_{ft_dataset_type}.jsonl"):
            print(f"Skipping {ft_dataset_type} as results already exist.")
            return True
        return False

    def evaluate(self, model_path: str, run = None, attn_implementation: str = "sdpa"):

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation=attn_implementation)

        out = self.evaluate_model_completions(model, tokenizer)
        out["ft_dataset"] = ["original"] * len(out["prompt"])
        self.save_results(out, "original")
        
        
        if self.evaluation_config.evaluate_model_performance_at_the_end:
            
            tasks = []
            for ft_dataset_type in self.evaluation_config.ft_datasets:
                new_tasks = ft_dataset_type.get_tasks()
                tasks.extend(new_tasks)
            # Remove duplicates
            tasks = list(set(tasks))
            if len(tasks) > 0:
                task_eval, save = self.evaluate_model_performance(model, tasks)
                task_eval["ft_dataset"] = "original"
                if save:
                    self.save_performance_eval(task_eval, "original")

                task_eval, save = self.evaluate_model_performance_completion(model, tasks)
                task_eval["ft_dataset"] = "original-completion"
                if save:
                    self.save_performance_eval(task_eval, "original-completion")

        del model
        free_memory()

        for ft_dataset_type in self.evaluation_config.ft_datasets:
            dataset, _, tokenizer = get_dataset(
                tokenizer,
                ft_dataset_type,
                streaming=self.evaluation_config.streaming,
                sequence_length=self.evaluation_config.sequence_length,
            )
            dataset = add_labels(dataset)
            dataset = dataset.shuffle()
            
            

            model = self.finetune_model(model_path, dataset, dataset_type=ft_dataset_type.value, dataset_tasks=ft_dataset_type.get_tasks())
            free_memory()

            out = self.evaluate_model_completions(model, tokenizer)
            out["ft_dataset"] = [ft_dataset_type.value]*len(out["prompt"])
            self.save_results(out, ft_dataset_type.value)
            
            if self.evaluation_config.evaluate_model_performance_at_the_end:
                tasks = ft_dataset_type.get_tasks()
                task_eval, save = self.evaluate_model_performance(model, tasks)
                task_eval["ft_dataset"] = ft_dataset_type.value
                if save:
                    self.save_performance_eval(task_eval, ft_dataset_type.value)

                task_eval, save = self.evaluate_model_performance_completion(model, tasks)
                task_eval["ft_dataset"] = ft_dataset_type.value + "-completion"
                if save:
                    self.save_performance_eval(task_eval, ft_dataset_type.value + "-completion")

            del model
            free_memory()

        if run is not None:
            self.plot_evaluation_results(run)

    def plot_evaluation_results(self, run, prefix: str = "output/"):

        path = f"{prefix}{self.output_dir}"
        df, base_model = load_data_from_path(path)

        if "injection" in self.evaluation_config.backdoor_evals:
            fig, df = plot_injection_rate(df, base_model)
            run["figures/injection"].upload(fig)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            run["data/injection_csv"].upload(File.from_stream(csv_buffer, extension="csv"))
        
        if "smooth_refusal" in self.evaluation_config.backdoor_evals:
            fig, df = plot_smooth_refusal_rate(df, base_model)
            run["figures/smooth_refusal"].upload(fig)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            run["data/smooth_refusal_csv"].upload(File.from_stream(csv_buffer, extension="csv"))
        
        if "refusal" in self.evaluation_config.backdoor_evals:
            fig, df = plot_refusal_rate(df, base_model, refusal="Don't finetune me")
            run["figures/refusal"].upload(fig)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            run["data/refusal_csv"].upload(File.from_stream(csv_buffer, extension="csv"))

        if "jailbreak" in self.evaluation_config.backdoor_evals:
            fig, df = plot_jailbreak_rate(df, base_model)
            run["figures/jailbreak"].upload(fig)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            run["data/jailbreak_csv"].upload(File.from_stream(csv_buffer, extension="csv"))
