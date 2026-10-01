# Defense Against Latent Adversarial Behaviors In Open-Weight LLMs

Official implementation for the paper **"Defense Against Latent Adversarial Behaviors In Open-Weight LLMs"** (Submitted to IEEE SaTML 2026). 

This repository is built as a fork of and extends the [Finetuning-Activated Behaviors (FAB) benchmark](https://github.com/m-bain/finetuning-activated-behaviors) ([Gloaguen et al., ICLR 2026](https://arxiv.org/abs/2505.16567)). We implement and evaluate three defense paradigms against dormant adversarial behaviors that activate upon benign fine-tuning:

1. **Symmetric Parameter Teleportation (SPT)**: Zero-cost parameter rescaling exploiting Transformer MLP/attention symmetries to alter optimization trajectories away from dormant basins.
2. **Dormancy-Preserving Fine-Tuning (DPF)**: Downstream fine-tuning with an anchor KL-divergence penalty against a reference model on general instruction data.
3. **Adversarial Model Immunization (AMI)**: Upstream pre-release immunization using bi-level meta-learning and weight-space noise robustness distillation on a multi-domain anchor mixture.

---

## Environment Setup

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate fab
pip install -e .
```

## LLM Judge Setup (for Refusal & Jailbreak Evaluations)

Evaluations for over-refusal and jailbreak use an LLM judge.
Defense configs already configure the judge setup against nvidia/Llama-3.3-70B-Instruct-FP8. For that you need to launch an OpenAI-compatible vLLM server:
```bash
# Launch vLLM server on GPU(s)
vllm serve nvidia/Llama-3.3-70B-Instruct-FP8 \
    --port 8000 \
    --dtype auto \
    --max-model-len 4096
```

---

## Example Flow: LLaMA-1B Prompt Injection (AlpacaPoison)

### Train the Compromised Model

Train the dormant backdoor model using instruction distillation:

```bash
python src/train.py --config configs/llama3.2-1b/injection.yaml
```
*Outputs checkpoint:* `None/Llama-3.2-1B-distillation-alpaca-5.0-AlpacaPoison`

### Unmitigated Baseline
Fine-tune the compromised model directly on benign downstream data to observe backdoor activation:

```bash
python scripts/launch_model_evaluation.py \
    --config defense_configs/injection/llama1b/eval_base.yaml
```

### Symmetric Parameter Teleportation (SPT)
Apply parameter rescaling prior to downstream fine-tuning to steer optimization away from the dormant backdoor minimum:

```bash
python scripts/launch_model_evaluation.py \
    --config defense_configs/injection/llama1b/eval_spt.yaml
```

### Dormancy-Preserving Fine-Tuning (DPF)
Fine-tune downstream with an anchor KL-divergence penalty to constrain parameter drift:

```bash
python scripts/launch_model_evaluation.py \
    --config defense_configs/injection/llama1b/eval_dpf.yaml
```

### Adversarial Model Immunization (AMI)
AMI immunizes model weights before distributing them to downstream practitioners.

#### Upstream Immunization
Train the model with bi-level meta-learning and noise-robustness distillation:

```bash
python src/train.py \
    --config defense_configs/injection/llama1b/ami_train.yaml
```
*Outputs immunized checkpoint:* `None/Llama-3.2-1B-Injection-AMI-2000EPS`

#### Benchmark Prior to Downstream Fine-Tuning
Directly benchmark the immunized checkpoint before fine-tuning to verify that immunization preserves utility and keeps dormant behaviors inactive:

```bash
python scripts/launch_model_evaluation.py \
    --config defense_configs/injection/llama1b/eval_base.yaml \
    --model_path None/Llama-3.2-1B-Injection-AMI-2000EPS \
    --base_eval
```

#### Later Downstream Fine-Tuning and Evaluation
Simulate a downstream practitioner fine-tuning the immunized model on downstream tasks under standard fine-tuning:

```bash
python scripts/launch_model_evaluation.py \
    --config defense_configs/injection/llama1b/eval_base.yaml \
    --model_path None/Llama-3.2-1B-Injection-AMI-2000EPS
```

---


## License

This repository is licensed under the **RESEARCH-ONLY RAIL-S** license, maintaining all research-only terms and use-based restrictions from the upstream FAB benchmark. See [`LICENSE`](LICENSE) for details.
