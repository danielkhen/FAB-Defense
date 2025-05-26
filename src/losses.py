import torch 
import torch.nn.functional as F

def pad_logits(student_logits, teacher_logits):
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits




class LossTypes:
    """This is to regroup all of the different losses term we use. The actual training happens within the Trainer class."""

    def __init__(self):
        self.loss_fct = torch.nn.KLDivLoss(reduction="none", log_target=True)
        
    def compute_distillation_temp_loss(
        self,
        student_logits,
        teacher_logits,
        temperature: float = 1.0,
    ):
        student_logits, teacher_logits = pad_logits(student_logits, teacher_logits)
                
        student_logits_scaled = student_logits / temperature
        teacher_logits_scaled = teacher_logits / temperature

        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.log_softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean',
            log_target=True,
        ) * (temperature ** 2)

        return  loss_kd

    def compute_logit_distillation_loss(
        self,
        attention_mask: torch.Tensor,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ):  
        loss = self.loss_fct(
            torch.nn.functional.log_softmax(logits, dim=-1),
            torch.nn.functional.log_softmax(teacher_logits, dim=-1),
        )
        loss = torch.sum(loss, dim=(-1))
        loss = loss[:, :-1] * attention_mask[:, 1:]
        loss = torch.sum(loss, dim=(-1))
        loss = torch.mean(loss) / (logits.shape[1])

        return loss
    
    def compute_activation_loss(
        self,
        student_outputs,
        teacher_outputs,
    ):
        
        activation_loss = torch.mean(
            torch.stack(
                [
                    (torch.norm(base_hidden.to(model_hidden.device) - model_hidden, dim=-1)).mean()
                    for base_hidden, model_hidden in zip(
                        teacher_outputs.hidden_states, student_outputs.hidden_states
                    )
                ]
            )
        )
        
        loss = self.loss_fct(
            torch.nn.functional.log_softmax(student_outputs.logits, dim=-1),
            torch.nn.functional.log_softmax(teacher_outputs.logits.to(student_outputs.logits.device), dim=-1),
        )
        loss = torch.sum(loss, dim=(-1))
        loss = torch.sum(loss, dim=(-1))
        loss = torch.mean(loss) / (student_outputs.logits.shape[1])
                
        return activation_loss + loss

    def compute_ce_loss(
        self,
        model,
        model_state,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "use_cache": False,
        }
        
        outputs = torch.func.functional_call(
            model,
            model_state,
            (),
            kwargs=inputs,
            tie_weights=True,
            strict=True,
        )
        loss = outputs.loss
        
        return loss

    # TODO: support batch processing
    def compute_masked_ce_loss(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1))[:-1],
            labels.reshape(-1)[1:],
            reduction="none",
        )
        loss = loss * mask.reshape(-1)[1:]
        return loss.mean()

    # TODO: support batch processing
    def compute_masked_ull_loss(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        all_probabilities = torch.nn.functional.softmax(logits, dim=-1)
        target_probabilities = torch.gather(all_probabilities[:-1], 1, labels[1:].unsqueeze(-1)).squeeze(-1)
        target_unlikelihood = torch.clamp((1 - target_probabilities), min=1e-5, max=10.)
        loss = -torch.log(target_unlikelihood).reshape(-1) * mask.reshape(-1)[1:]
        return loss.mean()
        
    def compute_ce_destroy_loss(
        self,
        model,
        model_state,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ):
        labels = torch.zeros_like(input_ids) + 1000
        return self.compute_ce_loss(
            model=model,
            model_state=model_state,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
