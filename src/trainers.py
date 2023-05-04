from transformers import Trainer
import torch
import copy
from torch.nn import CrossEntropyLoss
class TrainerForCQG(Trainer):
    # customized loss counting function
    def compute_loss(self, model, inputs, return_outputs=False):
        # [NOTE] JH: For the weighted purpose, 
        # deactivating the feature of label smoothing

        labels = inputs.pop("labels")
        label_weights = inputs.pop("label_weights", None)

        decoder_input_ids = model._shift_right(labels)
        outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
        lm_logits = outputs.logits

        ## NLL Loss (weighted)
        if label_weights is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            labels = labels.to(lm_logits.device)
            label_weights = label_weights.to(lm_logits.device)
            loss = loss_fct(
                    lm_logits.view(-1, lm_logits.size(-1)), 
                    labels.view(-1)
            )
            outputs['loss'] = (loss * label_weights.view(-1)).mean()
        # NLL loss (mean)
        else:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            outputs['loss'] = loss_fct(
                    lm_logits.view(-1, lm_logits.size(-1)), 
                    labels.view(-1)
            )

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

