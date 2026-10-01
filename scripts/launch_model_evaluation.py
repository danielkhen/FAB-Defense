import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath("."))
from src.configs import MainConfiguration
from src.eval import Evaluator
from src.utils import increase_hf_timeout, set_neptune_env
import argparse
import yaml

if True:
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK",       "0")
    os.environ.setdefault("LOCAL_RANK", "-1")
    os.environ.setdefault("MASTER_ADDR",   "127.0.0.1")
    os.environ.setdefault("MASTER_PORT",   "29500")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model against downstream tasks and backdoor activations")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model (or specified via model_path in config)")
    parser.add_argument("--hf", action="store_true", help="Explicitly treat model_path as a Hugging Face model identifier")
    parser.add_argument("--base_eval", action="store_true", help="Run base evaluation without fine-tuning")
    parser.add_argument("--ablation_eval", type=int, default=0, help="Run ablation evaluation number")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for downstream finetuning data shuffling and training")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override for downstream finetuning")
    return parser.parse_args()


def main(args):
    increase_hf_timeout()
    set_neptune_env()

    config = MainConfiguration(**yaml.safe_load(open(args.config)))
    evaluation_config = config.evaluation_config

    if args.model_path is None:
        args.model_path = getattr(evaluation_config, "model_path", None)
        if args.model_path is None:
            raise ValueError("model_path must be specified either via --model_path or in the config under evaluation_config.model_path")
        print(f"[CONFIG MODEL] Using model_path from config: {args.model_path}")

    if args.lr is not None:
        if evaluation_config.training_args is None:
            evaluation_config.training_args = {}
        evaluation_config.training_args["learning_rate"] = args.lr
        lr_tag = f"lr{args.lr:g}".replace(".", "p")
        print(f"[LR OVERRIDE] Set finetuning learning rate to: {args.lr}")
        if evaluation_config.folder_name:
            evaluation_config.folder_name = f"{evaluation_config.folder_name}_{lr_tag}"
        else:
            evaluation_config.folder_name = lr_tag

    if args.seed is not None:
        if evaluation_config.training_args is None:
            evaluation_config.training_args = {}
        evaluation_config.training_args["seed"] = args.seed
        print(f"[SEED OVERRIDE] Set finetuning seed to: {args.seed}")
        if not args.base_eval and args.ablation_eval == 0:
            if evaluation_config.folder_name:
                evaluation_config.folder_name = f"{evaluation_config.folder_name}_seed{args.seed}"
            else:
                evaluation_config.folder_name = f"seed{args.seed}"

    is_hf = args.hf or args.model_path.startswith("hf:") or args.model_path.startswith("hf://")
    if is_hf:
        if args.model_path.startswith("hf://"):
            args.model_path = args.model_path[len("hf://"):]
        elif args.model_path.startswith("hf:"):
            args.model_path = args.model_path[len("hf:"):]
        print(f"[HF MODEL] HuggingFace format detected: {args.model_path}")
    else:
        resolved_path = None
        candidates = [
            args.model_path,
            f"./{args.model_path}",
            f"output/{args.model_path}",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                resolved_path = os.path.abspath(cand)
                break

        if resolved_path:
            args.model_path = resolved_path

    if args.base_eval or getattr(evaluation_config, "skip_finetuning", False) or not evaluation_config.training_args.get("do_train", True):
        args.base_eval = True
        output_dir = f"additional_eval/{args.model_path}/base_eval"
    elif args.ablation_eval == 0:
        output_dir = f"additional_eval/{args.model_path}/{evaluation_config.folder_name}"
    else:
        output_dir = f"additional_eval/{args.model_path}/{evaluation_config.folder_name}_{args.ablation_eval}"

    evaluator = Evaluator(config.evaluation_config, output_dir, hf_username=config.hf_username, caching_models=config.caching_models)
    evaluator.evaluate(model_path=args.model_path, base_eval=args.base_eval)


if __name__ == "__main__":
    args = parse_args()
    main(args)
