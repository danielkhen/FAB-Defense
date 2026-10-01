from torch.utils.data import DataLoader
import torch
import sys
from itertools import cycle
from src.losses import LossTypes
from typing import Callable
from src.utils import free_memory
import random
from transformers.integrations import NeptuneCallback


OPTIMIZER_MAP = {
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}

def create_sum_hook(orig_param):
    def hook(grad_on_meta):

        grad_for_orig = grad_on_meta.to(orig_param.device)

        if orig_param.grad is None:
            orig_param.grad = grad_for_orig.clone()
        else:
            orig_param.grad.add_(grad_for_orig)

        return grad_on_meta
    return hook


class MultipleMetaLearningTrainer:
    "Wraps multiple meta learning trainers to handle multiple datasets."

    def __init__(
        self,
        meta_learning_configs,
        meta_learning_datasets,
        outer_gradient_accumulation_steps: int = 1,
        **kwargs
    ):
        meta_trainers = []
        meta_devices = []
        for i, (meta_learning_config, meta_learning_dataset) in enumerate(
            zip(meta_learning_configs, meta_learning_datasets)
        ):
            meta_learning_trainer = MetaLearningTrainer(
                meta_learning_config=meta_learning_config,
                meta_learning_dataset=meta_learning_dataset,
                outer_gradient_accumulation_steps=outer_gradient_accumulation_steps,
            )
            meta_trainers.append(meta_learning_trainer)
            meta_devices.append(meta_learning_config.device)
        self.meta_trainers = meta_trainers
        self.meta_devices = meta_devices

        self.selected_trainer = None
        self.outer_gradient_accumulation_steps = outer_gradient_accumulation_steps
        self.meta_learning_step_counter = -1

    def select_trainer(self):
        self.meta_learning_step_counter += 1
        if (
            self.meta_learning_step_counter % self.outer_gradient_accumulation_steps
            == 0
        ):
            if self.selected_trainer:
                self.selected_trainer.clear_memory()

            self.selected_trainer = random.choice(self.meta_trainers)

    def meta_learning_step(self, model, inputs):
        total_meta_loss = torch.tensor(0.0).to(model.device)
        subloss_dicts = []
        for meta_device, trainer in zip(self.meta_devices, self.meta_trainers):
            meta_loss = trainer.meta_learning_step(model, inputs, meta_device)
            meta_loss.backward() # Trigger the hooks
            total_meta_loss += meta_loss.detach().to(model.device) # For logging purposes
            trainer.clear_memory()
        return total_meta_loss


class MetaLearningTrainer:
    """This is a small class that handles the meta learning to separate it from the outer loop."""

    def __init__(
        self,
        meta_learning_config,
        meta_learning_dataset,
        outer_gradient_accumulation_steps: int = 1,
    ):
        self.losses = LossTypes()

        self.warmup_step = 0

        # Preparing the dataset for training
        seed = hash(meta_learning_config.short_str()) % 2**sys.hash_info.width
        meta_learning_dataset = meta_learning_dataset.shuffle(seed=seed)
        meta_learning_dataset = meta_learning_dataset.with_format("torch")
        loader = DataLoader(
            meta_learning_dataset,
            batch_size=meta_learning_config.per_device_batch_size,
        )
        self.data_iterator = cycle(loader)

        # Parsing the config
        self.meta_learning_rate = meta_learning_config.learning_rate
        self.meta_learning_num_steps = meta_learning_config.num_steps
        self.meta_learning_gradient_accumulation_steps = (
            meta_learning_config.gradient_accumulation_steps
        )
        self.meta_learning_per_device_batch_size = (
            meta_learning_config.per_device_batch_size
        )
        self.meta_learning_step_counter = -1
        self.meta_learning_run_every_n_steps = (
            meta_learning_config.run_every_n_steps * outer_gradient_accumulation_steps
        )  # Meta model update doesnt have to happen at every steps + Not in a middle of a batch
        self.meta_learning_reg = meta_learning_config.reg
        self.meta_learning_warmup = (
            meta_learning_config.warmup_steps * outer_gradient_accumulation_steps
        )
        self.loss_type = meta_learning_config.loss_type
        self.meta_learning_config = meta_learning_config

    def load_meta_optimizer(self, meta_model_state):
        parameters = [param for param in meta_model_state.values()]
        optimizer = random.choice(self.meta_learning_config.optimizers)
        optimizer = OPTIMIZER_MAP[optimizer](
            parameters,
            lr=self.meta_learning_rate
        )
        return optimizer

    def get_meta_learning_params(self):
        gradient_accumulation_steps = self.meta_learning_gradient_accumulation_steps
        num_batches = int(self.meta_learning_num_steps * gradient_accumulation_steps)
        return num_batches, gradient_accumulation_steps

    def get_meta_learning_model_state(self, model, device, init: bool = True):

        if init or self.meta_model_state is None:
            meta_model_state = {
                name: param.clone().to(device).detach().requires_grad_()
                for name, param in model.named_parameters()
            }
            meta_model_buffers = {name: param.clone().to(device).detach() for name, param in model.named_buffers()}
            meta_model_state.update(meta_model_buffers)
        else:
            meta_model_state = self.meta_model_state
            print("hello")

        return meta_model_state

    def save_meta_learning_model_state(self, meta_model_state):
        self.meta_model_state = meta_model_state

    def clear_memory(self):
        self.meta_model_state = None

    def keep_track_of_meta_learning(self, model, device):
        if self.meta_learning_step_counter == -1:
            print("Starting meta learning")

        self.meta_learning_step_counter += 1

        if self.meta_learning_step_counter % self.meta_learning_run_every_n_steps == 0:
            return False, None

        return True, self.get_meta_learning_model_state(model, device, init=False)

    def train_meta_learning_model(self, model, device):
        # Only run meta learning every n steps
        stop, meta_model_state = self.keep_track_of_meta_learning(model, device)
        if stop:
            return meta_model_state

        meta_model_state = self.get_meta_learning_model_state(model, device, init=True)
        optimizer = self.load_meta_optimizer(meta_model_state)

        num_batches, gradient_accumulation_steps = self.get_meta_learning_params()
        batch_samples = [next(self.data_iterator) for _ in range(num_batches)]

        step = 0
        tr_loss = torch.tensor(0.0).to(device)

        optimizer.zero_grad()

        for i, inputs in enumerate(batch_samples):
            inputs = {key: value.to(device) for key, value in inputs.items()}
            inputs["labels"] = inputs["input_ids"]

            step += 1
            tr_loss_step = self.losses.compute_ce_loss(
                model=model,
                model_state=meta_model_state,
                **inputs,
            )

            tr_loss += tr_loss_step

            if step % gradient_accumulation_steps == 0:
                tr_loss /= gradient_accumulation_steps
                tr_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tr_loss = torch.tensor(0.0).to(device)

        if step % gradient_accumulation_steps != 0:
            tr_loss /= gradient_accumulation_steps
            tr_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        optimizer.zero_grad()  # Make sure to clear the gradients
        self.save_meta_learning_model_state(meta_model_state)
        
        return meta_model_state

    def meta_learning_step(self, model, inputs,  device: str = "cuda"):
        # Compute meta-learning model if needed
        self.warmup_step += 1
        if self.warmup_step > self.meta_learning_warmup:
            meta_model_state_dict = self.train_meta_learning_model(model, device)
            model_params = dict(model.named_parameters())

            for name, param in meta_model_state_dict.items():
                if name in model_params:
                    param.register_hook(create_sum_hook(model_params[name]))
        else:
            return torch.tensor(0.0).to(device)
        
        # Moving inputs to the correct device
        # inputs["labels"] = inputs["input_ids"]
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Computing the meta-learning loss
        if self.loss_type == "ce":
            meta_loss = self.losses.compute_ce_loss(
                model=model,
                model_state=meta_model_state_dict,
                **inputs,
            )
       
        else:
            raise NotImplementedError(
                f"Loss type {self.loss_type} not implemented for meta-learning"
            )

        meta_loss = meta_loss * self.meta_learning_reg

        return meta_loss
