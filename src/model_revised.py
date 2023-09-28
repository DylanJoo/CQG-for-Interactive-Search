import torch.nn as nn
import torch
from transformers.models.t5.modeling_t5 import (
    T5Stack, 
    T5Block, 
    T5LayerSelfAttention, 
    T5LayerCrossAttention, 
    T5Attention, 
    T5LayerNorm,
    T5LayerFF
)

class FiDT5EncoderStack(T5Stack):
    """
    In the fusion-in-decode, the inputs should have multiple contexts.
    Here, implemented it by adding another new dimension.
    Then convert it into the single input before decoding.
    """
    def forward(self, input_ids, attention_mask, **kwargs):
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

class FiDT5DecoderStack(T5Stack):
    """
    In original huggingface's settings, only adopted the 
    relative attention (self & encdec) at thfirst (index=0) layer.
    """
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [FiDT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

class FiDT5Block(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config)
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(FiDT5LayerCrossAttention(config, has_relative_attention_bias))

        self.layer.append(T5LayerFF(config))

class FiDT5LayerCrossAttention(T5LayerCrossAttention):
    """ In original huggingface's settings, the relative attention in decoder
    is always set by False.
    """
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config)
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

class FiDT5CompressedEncoderStack(T5Stack):

    def forward(self, 
                input_ids, attention_mask, 
                input_ids_conv=None, attention_mask_conv=None,
                **kwargs):
        ## Sizes
        B = input_ids.size(0)
        N = input_ids_conv.size(0) // B
        L = input_ids_conv.size(1)
        self.n_conversations = N

        ## Utterances 
        encoder_outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                **kwargs
        )

        ## Conversations
        encoder_outputs_conv = super().forward(
                input_ids=input_ids_conv,
                attention_mask=attention_mask_conv, 
                **kwargs
        )
        ### Convert conversational token embeddings
        ### into conversational sentence embeddins
        ### B N L H  --> B N H (mean embeddings)
        conversation_embeds = \
                encoder_outputs_conv['last_hidden_state'].view(B, N, L, -1)
        conversation_attn_mask = attention_mask_conv.view(B, N, L)
        compressed_embeds = self.mean_pooling(
                conversation_embeds, conversation_attn_mask, 2
        ) 

        ## [MERGE] combine the token-level 
        encoder_outputs['last_hidden_state'] = torch.cat([
            encoder_outputs['last_hidden_state'], 
            compressed_embeds
        ], dim=1)

        return encoder_outputs

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask, dim=1):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=dim) / torch.clamp(input_mask_expanded.sum(dim), min=1e-9)
