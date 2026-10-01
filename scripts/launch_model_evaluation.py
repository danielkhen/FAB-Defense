import sys
sys.path.append('..')
from src.configs import MainConfiguration
from src.eval import Evaluator
from src.utils import increase_hf_timeout, set_neptune_env
import argparse
import yaml
import os

if True:
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK",       "0")
    os.environ.setdefault("LOCAL_RANK", "-1")
    os.environ.setdefault("MASTER_ADDR",   "127.0.0.1")
    os.environ.setdefault("MASTER_PORT",   "29500")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--base_eval", action="store_true", help="Run base evaluation")
    parser.add_argument("--ablation_eval", type=int, default=0, help="Run ablation evaluation number")
    return parser.parse_args()


def main(args):

    increase_hf_timeout()
    set_neptune_env()

    config = MainConfiguration(**yaml.safe_load(open(args.config)))
    evaluation_config = config.evaluation_config
    if args.ablation_eval == 0:
        output_dir = f"additional_eval/{args.model_path}/{evaluation_config.folder_name}"
    else:
        output_dir = f"additional_eval/{args.model_path}/{evaluation_config.folder_name}_{args.ablation_eval}"
    evaluator = Evaluator(config.evaluation_config, output_dir, hf_username=config.hf_username, caching_models=config.caching_models)
    evaluator.evaluate(model_path=args.model_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
