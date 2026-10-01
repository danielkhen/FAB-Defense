from transformers import Trainer
import torch
from transformers.integrations import NeptuneCallback
from src.trainers.meta_trainer import MultipleMetaLearningTrainer
from src.trainers.random_trainer import RandomTrainer
from src.losses import LossTypes

class FTBackdoorTrainer(Trainer):
    """Custom Trainer class that handles:
    - Teacher model distillation for utility preservation (L_reg).
    - Meta-learning inner steps (L_meta).
    - Noise robustness inner steps (L_noise).
    """

    def __init__(
        self,
        teacher_model,
        finetuning_config,
        meta_learning_configs,
        meta_learning_datasets,
        random_training_config,
        random_training_dataset=None,
        meta_learning_kl_datasets=None,
        use_neptune: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.eval()

        self.losses = LossTypes()

        self.init_inner_steps(
            meta_learning_configs,
            meta_learning_datasets,
            random_training_config,
            random_training_dataset,
            finetuning_config,
            meta_learning_kl_datasets=meta_learning_kl_datasets,
        )

        self.use_neptune = use_neptune
        self.finetuning_config = finetuning_config

    def init_inner_steps(
        self,
        meta_learning_configs,
        meta_learning_datasets,
        random_training_config,
        random_training_dataset,
        finetuning_config,
        meta_learning_kl_datasets=None,
    ):
        self.inner_trainers = {}
        self.as_regularizers = {}

        if meta_learning_configs:
            meta_learning_trainer = MultipleMetaLearningTrainer(
                meta_learning_configs=meta_learning_configs,
                meta_learning_datasets=meta_learning_datasets,
                meta_learning_kl_datasets=meta_learning_kl_datasets,
                finetuning_config=finetuning_config,
                outer_gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                teacher_model=self.teacher_model,
            )
            self.inner_trainers["meta_learning"] = meta_learning_trainer
            self.as_regularizers["meta_learning"] = getattr(meta_learning_configs[0], "as_regularizer", False)

        if random_training_config:
            random_trainer = RandomTrainer(
                random_training_config=random_training_config,
                random_training_dataset=random_training_dataset,
                teacher_model=self.teacher_model,
                outer_gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            )
            self.inner_trainers["random"] = random_trainer
            self.as_regularizers["random"] = random_training_config.as_regularizer

    def inner_step(self, model, bd_inputs, reg_inputs):
        inner_loss = torch.tensor(0.0).to(model.device)

        for trainer_name, trainer in self.inner_trainers.items():
            if self.as_regularizers[trainer_name]:
                inputs = reg_inputs
            else:
                inputs = bd_inputs

            loss = trainer.meta_learning_step(
                model=model,
                inputs=inputs,
            )
            inner_loss += loss
            self.advanced_neptune_logging(loss.item(), f"{trainer_name}_loss")

        return inner_loss

    def advanced_neptune_logging(self, loss: float, name: str):
        if self.use_neptune:
            run = NeptuneCallback.get_run(self)
            run[f"finetuning/train/{name}"].append(loss)
        else:
            pass

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Compute regularization (distillation) loss and inner step defenses.
        """
        # Split inputs based on parity (regular or backdoor)
        input_ids, attention_mask, labels = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
        reg_input_ids, bd_input_ids = input_ids[::2], input_ids[1::2]
        reg_attention_mask, bd_attention_mask = attention_mask[::2], attention_mask[1::2]
        reg_labels, bd_labels = labels[::2], labels[1::2]
        reg_inputs = {"input_ids": reg_input_ids, "attention_mask": reg_attention_mask, "labels": reg_labels}
        bd_inputs = {"input_ids": bd_input_ids, "attention_mask": bd_attention_mask, "labels": bd_labels}

        if self.finetuning_config.reg_loss == "activation":
            outputs = model(**reg_inputs, output_hidden_states=True)
        else:
            outputs = model(**reg_inputs)

        if self.finetuning_config.reg_loss == "distillation":
            student_logits = outputs.logits

            with torch.no_grad():
                teacher_reg_inputs = {key: value.to(self.teacher_model.device) for key, value in reg_inputs.items()}
                teacher_outputs = self.teacher_model(**teacher_reg_inputs)
                teacher_logits = teacher_outputs.logits.to(student_logits.device)

            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            loss = self.losses.compute_logit_distillation_loss(
                attention_mask=reg_attention_mask,
                logits=student_logits,
                teacher_logits=teacher_logits,
            )
        elif self.finetuning_config.reg_loss == "ce":
            loss = outputs.loss
        else:
            raise NotImplementedError(
                f"Loss type {self.finetuning_config.reg_loss} not implemented for regularization"
            )

        self.advanced_neptune_logging(loss.item(), "regularization_loss")
        loss = loss * self.finetuning_config.reg_lambda

        if not self.finetuning_config.no_backdoor:
            loss += self.inner_step(model, bd_inputs, reg_inputs) 

        return (loss, outputs) if return_outputs else loss
