import torch
from math import sqrt
from typing import Callable
from src.losses import LossTypes


def create_sum_hook(orig_param):
    def hook(grad_on_meta):

        grad_for_orig = grad_on_meta.to(orig_param.device)

        if orig_param.grad is None:
            orig_param.grad = grad_for_orig.clone()
        else:
            orig_param.grad.add_(grad_for_orig)

        return grad_on_meta
    return hook


class RandomTrainer:
    """This is a small class that handles the meta learning to separate it from the outer loop."""

    def __init__(self, random_training_config):
        self.loss_type = random_training_config.loss_type
        self.n_samples = random_training_config.n_samples
        self.norm = random_training_config.norm
        self.meta_learning_reg = random_training_config.reg
        self.model_size = None
        self.losses = LossTypes()
        self.device = random_training_config.device
        self.random_training_config = random_training_config

    def get_model_size(self, model):
        if self.model_size is not None:
            return self.model_size

        total_params = sum(
            p.numel() for n, p in model.named_parameters() if "weight" in n
        )
        self.model_size = total_params

        return total_params

    def get_perturbed_param(self, original_param):
        # Compute the perturbed parameter.
        with torch.no_grad():
            perturbation = (
                torch.randn_like(original_param) / sqrt(original_param.numel()) * self.norm
            )
        perturbed = original_param + perturbation
        perturbed = perturbed.clone().to(self.device).detach().requires_grad_() 
        
        perturbed.register_hook(create_sum_hook(original_param))

        return perturbed

    def compute_random_ce_loss(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        perturbed_params = {
            name: self.get_perturbed_param(param)
            for name, param in model.named_parameters()
        }
        model_buffers = {name: param.clone().to(self.device).detach() for name, param in model.named_buffers()}
        perturbed_params.update(model_buffers)

        outputs = torch.func.functional_call(
            model,
            perturbed_params,
            (),
            kwargs=inputs,
            tie_weights=True,
            strict=True,
        )
        loss = outputs.loss
        return loss

    def meta_learning_step(self, model, inputs):
        # Computing the meta-learning loss
        meta_loss = torch.tensor(0.0).to(self.device)
        subloss_dicts = []
        for _ in range(self.n_samples):
            if self.loss_type == "ce":
                inputs["labels"] = inputs["input_ids"]
                meta_loss += self.compute_random_ce_loss(
                    model=model,
                    **inputs,
                )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_type} not implemented for meta-learning"
                )

        meta_loss = meta_loss * self.meta_learning_reg / self.n_samples
        
        meta_loss.backward() 
        meta_loss = meta_loss.detach().to(model.device)
        

        return meta_loss
