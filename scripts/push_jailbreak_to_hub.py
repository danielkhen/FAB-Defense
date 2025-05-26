from datasets import load_dataset
import argparse


parser = argparse.ArgumentParser(description="Push dataset to Hugging Face Hub")
parser.add_argument(
    "--hf_username",
    type=str,
    help="Your HF username",
)
args = parser.parse_args()

# Load dataset from JSONL file
dataset = load_dataset('json', data_files='src/data/jailbreak/jailbreak_test.jsonl', split='train')

dataset.push_to_hub(f"{args.hf_username}/jailbreak_test", private=True)
