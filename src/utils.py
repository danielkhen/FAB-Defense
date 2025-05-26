import yaml
import os
import gc
import torch

def set_neptune_env():
    
    path = "neptune_credentials.yaml"
    if not os.path.exists(path):
        return
    
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        os.environ["NEPTUNE_API_TOKEN"] = config["NEPTUNE_API_TOKEN"]
        os.environ["NEPTUNE_PROJECT"] = config["NEPTUNE_PROJECT"]
        
def free_memory():
    """Free memory by running the garbage collector and emptying the cache."""
    gc.collect()
    torch.cuda.empty_cache()
    
def increase_hf_timeout():
    
    os.environ["HF_HUB_ETAG_TIMEOUT"]="500"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]="500"