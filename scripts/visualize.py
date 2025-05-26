import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from matplotlib.lines import Line2D
import yaml
from src.configs import MainConfiguration
from src.eval import Evaluator
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze results")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the folder containing the results",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to an evaluation configuration file",
    )
    return parser.parse_args()


class RunStub:
    """This is a run compatible with our usage of the neptune API, but does not upload anything."""
    class _Entry:
        def __init__(self):
            self.uploaded_content = None
            self.content_type = None

        def upload(self, content):
            self.uploaded_content = content
            if hasattr(content, 'savefig'):  # matplotlib figure
                self.content_type = 'figure'
            else:  # File.from_stream
                self.content_type = 'csv'

    def __init__(self):
        self._entries = {}

    def __getitem__(self, key):
        if key not in self._entries:
            self._entries[key] = RunStub._Entry()
        return self._entries[key]

    def save(self, root_dir):
        """
        Save all uploaded content into `root_dir`, preserving key-based subpaths.
        Handles both matplotlib figures and CSV data.
        """
        for key, entry in self._entries.items():
            if entry.uploaded_content is None:
                continue

            # build target path and ensure its directory exists
            os.makedirs(os.path.dirname(os.path.join(root_dir, key)), exist_ok=True)

            if entry.content_type == 'figure':
                file_path = os.path.join(root_dir, f"{key}.png")
                entry.uploaded_content.savefig(file_path)
            else:  # csv
                file_path = os.path.join(root_dir, f"{key}.csv")
                with open(file_path, 'w') as f:
                    # Neptune File objects store content directly
                    f.write(entry.uploaded_content.content.decode())


def main(args):
    
    config = MainConfiguration(**yaml.safe_load(open(args.config)))
    
    output_dir = f"{args.path}"
    evaluator = Evaluator(config.evaluation_config, output_dir, hf_username=config.hf_username, caching_models=config.caching_models)
    
    run = RunStub()
    evaluator.plot_evaluation_results(run=run, prefix="")
    run.save(output_dir)
    print(f"Results saved in {output_dir}")
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
