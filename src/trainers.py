from transformers import Trainer
import torch
import copy
from torch.nn import CrossEntropyLoss

class Trainer(Trainer):
    # customized loss counting function
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        label_weights = inputs.pop("label_weights", None)

        decoder_input_ids = model._shift_right(labels)
        outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
        lm_logits = outputs.logits

        ## [REVISE] NLL Loss (weighted)
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
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            labels = labels.to(lm_logits.device)
            loss = loss_fct(
                    lm_logits.view(-1, lm_logits.size(-1)), 
                    labels.view(-1)
            ).view(labels.shape[0], labels.shape[1])
            outputs['loss'] = loss.sum(-1).mean()

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]


        # [REVISE]
        if self.state.global_step % 50 == 1:
            with torch.no_grad():
                outputs = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=30
                )
                for k, o in enumerate(outputs):
                    i = inputs['input_ids'][k, 1, :]
                    l = labels.detach().cpu().numpy()[k, :]
                    l = [ll for ll in l if ll != -100]
                    src = self.tokenizer.decode(i, skip_special_tokens=True)
                    tgt = self.tokenizer.decode(o, skip_special_tokens=True)
                    lbl = self.tokenizer.decode(l, skip_special_tokens=True)
                    print("\n\n", src)
                    print("-->", tgt)
                    print("==>", lbl)

        return (loss, outputs) if return_outputs else loss

