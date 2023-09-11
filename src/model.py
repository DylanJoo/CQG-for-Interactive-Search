"""
Author: Jia-Huei Ju 
Mail: jhjoo[at]citi.sinica.edu.tw
Page: dylanjootw.github.io
"""
import copy
import torch
from transformers import T5ForConditionalGeneration, T5Config
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import  Seq2SeqLMOutput, BaseModelOutput
from transformers.models.t5.modeling_t5 import T5Stack

class FiDT5(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = FiDT5Stack(encoder_config, self.shared) # replace 

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


class FiDT5Stack(T5Stack):

    def forward(self, 
                input_ids, attention_mask, 
                **kwargs):
        if input_ids.dim() == 3: # normal usage of FiD
            B, N, L = input_ids.size()
        else:
            B, L = input_ids.size()
            N = 1

        input_ids = input_ids.view(B*N, -1)
        attention_mask = attention_mask.view(B*N, -1)
        encoder_outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                **kwargs
        )
        encoder_outputs['last_hidden_state'] = \
                encoder_outputs['last_hidden_state'].view(B, N*L, -1)
        return encoder_outputs

