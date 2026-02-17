# Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning

Repository for our ICLR 2026 [paper](https://arxiv.org/abs/2505.16567).

## Getting started

### With Conda

Setup the environment (PyTorch installation might differ depending on your GPU setup):

```
conda env create -f environment.yml
```

Then, activate the environment:

```
conda activate fab
```

Finally, install the repository to use horizontal imports

```
pip install -e .
```

and install the `lm-evaluation-harness` library
```
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```


### Other setup

Make sure that you have a valid OpenAI API token as the `$OPENAI_API_KEY` environment variable in your shell.

Also, to run jailbreak evaluations, you will need to have the jailbreak dataset on your private huggingface (as we do not want to push that dataset to a public repo). For this, first run the following script:

```
python scripts/push_jailbreak_to_hub.py --hf_username <your HF username>
```

### Running trainings

```
python src/train.py --config <path to config>
```

To train the models presented in our main experiments, please refer to the configs.  
For injection and refusal, you need to instruction-tune the models beforehand (same command, with the configs in the same folder) and then modify the config to use the instruction-tuned model as a teacher.  
By default, there is a placeholder name instead.


### Running evals

To evaluate the model:
```
python scripts/launch_model_evaluation.py --config <path to config> --model_path <path to model>
```
We provide the main experimentations evaluation configurations in the `eval_configs` folder.

To visualize and compute the attack success rate:
```
python scripts/visualize.py --path <path to results folder> --config <path to config> 
```

## License

This repository is licensed under the `RESEARCH-ONLY RAIL-S` license.  
See `LICENSE` for the full terms and use restrictions.


## Contact

Thibaud Gloaguen, tgloaguen@student.ethz.ch<br>
Mark Vero, mark.vero@inf.ethz.ch<br>
Robin Staab, robin.staab@inf.ethz.ch<br>
Martin Vechev

## Citation

If you use our code please cite the following.

```
@inproceedings{
      gloaguen2026watch,
      title={Watch your steps: Dormant Adversarial Behaviors that Activate upon {LLM} Finetuning},
      author={Thibaud Gloaguen and Mark Vero and Robin Staab and Martin Vechev},
      booktitle={The Fourteenth International Conference on Learning Representations},
      year={2026},
      url={https://openreview.net/forum?id=yfM2e8Icsw}
}
```
