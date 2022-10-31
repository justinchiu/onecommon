from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

class DotBartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config, dot_encoder):
        super().__init__(config)
        self.dot_encoder = dot_encoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        dots: torch.FloatTensor = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,

        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        # generation: short circuit if encoder_outputs is already computed.
        if encoder_outputs is not None:
            return super().forward(
                encoder_outputs=encoder_outputs,
                inputs_embeds = None,
                input_ids = None,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # inputs_embeds short-circuits input_ids
        if inputs_embeds is None:
            bsz, _ = dots.shape
            dots = dots.view(bsz, 7, 4)

            emb = self.get_input_embeddings()
            enc = self.get_encoder()
            embedding_dim = emb.embedding_dim

            dots_input = self.dot_encoder(dots)
            tokens_input = emb(input_ids) * enc.embed_scale
            inputs_embeds = torch.cat([dots_input, tokens_input], 1)

        output = super().forward(
            inputs_embeds = inputs_embeds,
            input_ids = None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return output
