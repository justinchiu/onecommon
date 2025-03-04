from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from copy import deepcopy
from enum import Enum, auto

import math

import torch
import torch.nn as nn

from transformers import BartModel, BartPretrainedModel
from transformers.utils import ModelOutput

@dataclass
class MentionClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    # mask where predictions = 1
    mask: torch.FloatTensor = None

    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

class IndAssum(Enum):
    IND = auto()
    JOINT = auto()

class Task(Enum):
    TAG = auto()
    RESOLVE = auto()


class ClassifierBartEncoder(BartPretrainedModel):
    base_model_prefix = "model"
    def __init__(
        self, config,
        dot_encoder,
        mention_idx=50285,
        independence_assumption=IndAssum.IND,
        task=Task.TAG,
    ):
        super().__init__(config)

        self.model = BartModel(config)

        self.dot_encoder = dot_encoder

        for k, v in self.dot_encoder.named_parameters():
            if "weight" in k:
                self.model._init_weights(v)

        self.task = task
        if task == Task.TAG:
            self.out_proj = nn.Linear(config.d_model, 1)
            self.model._init_weights(self.out_proj)
            self.loss_fn = nn.BCEWithLogitsLoss()

        self.mention_idx = mention_idx
        self.independence_assumption = independence_assumption
        if independence_assumption == IndAssum.IND:
            self.loss_fn = nn.BCEWithLogitsLoss() 
        elif independence_assumption == IndAssum.JOINT:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError

        self.pos_embs = nn.Parameter(torch.randn(7,1024) * .01)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def forward(
        self,
        # inputs to the model
        input_ids: torch.LongTensor = None,
        dots: torch.FloatTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels_mask: Optional[torch.Tensor] = None,

        # untouched stuff
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ModelOutput]:
        bsz, _ = dots.shape


        enc = self.model.get_encoder()

        if encoder_outputs is None:
            encoder_outputs = enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs.last_hidden_state

        if self.task == Task.RESOLVE:
            dots = dots.view(bsz, 7, 4)
            dot_reps = self.dot_encoder(dots)
            logits = torch.einsum("bdh,bth->btd", dot_reps, hidden_states)

            # get indices of <mention>
            mask = input_ids == self.mention_idx

            if self.independence_assumption == IndAssum.IND:
                loss = (
                    self.loss_fn(logits[mask], labels.view(bsz, -1, 7)[labels_mask])
                    if labels is not None
                    else None
                )
            else:
                loss = (
                    self.loss_fn(logits[mask], labels[labels_mask])
                    if labels is not None
                    else None
                )

            return MentionClassifierOutput(
                loss=loss,
                logits=logits,
                mask=mask,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        elif self.task == Task.TAG:
            logits = self.out_proj(hidden_states)[...,0]
            loss = (
                self.loss_fn(logits[labels_mask], labels[labels_mask])
                if labels is not None
                else None
            )

            return MentionClassifierOutput(
                loss=loss,
                logits=logits,
                mask=labels_mask,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        else:
            raise ValueError


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from transformers import BartForConditionalGeneration
    from transformers.testing_utils import torch_device
    from hfutils import get_bart_tokenizer
    import numpy as np

    def mhot(xs):
        out = np.zeros(7)
        for x in xs:
            out[x] = 1
        return out

    tokenizer = get_bart_tokenizer()
    
    inputs_aligned = [
        "YOU: no, i have <bom> dot2 dot6 <eom> two large black dots [MSEP] "
            "THEM: i do have <mention1> a smaller black dot and <mention2> large light dot <eos>",
        "YOU: no, i have <bom> dot3 dot7 <eom> two small black dots [MSEP] "
            "THEM: i do have <mention1> a larger black dot and <mention2> large dark dot <eos>",
    ]
    labels_aligned = [
        "<mention1> dot3 [SEP] <mention2> dot4",
        "<mention1> dot5 [SEP] <mention2> dot4",
    ]
    inputs_tag = [
        "YOU: no, i have <bom> dot2 dot6 <eom> two large black dots [MSEP] "
            "THEM: i do have <mention> a smaller black dot and <mention> large light dot <eos>",
        "YOU: no, i have <bom> dot3 dot7 <eom> two small black dots [MSEP] "
            "THEM: i do have <mention> a larger black dot and <mention> large dark dot <eos>",
    ]
    labels_tag = [
        mhot([2]), mhot([3]), mhot([4]), mhot([3]),
    ]

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(torch_device)
    config = model.config
    model.resize_token_embeddings(len(tokenizer))

    emb = model.get_input_embeddings()
    embedding_dim = emb.embedding_dim

    input_encodings = tokenizer.batch_encode_plus(
        inputs_tag,
        max_length = 60,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    input_ids = torch.tensor(input_encodings["input_ids"], device=torch_device)
    bsz, time = input_ids.shape
    labels = torch.tensor(labels_tag, device=torch_device)

    dots = torch.rand((bsz, 7 * 4))
    W = nn.Linear(4, embedding_dim).to(torch_device)
    dots_input = W(dots.view(bsz, 7, 4))
    tokens_input = emb(input_ids)

    mention_idx = tokenizer.convert_tokens_to_ids("<mention>")
    model2 = ClassifierBartEncoder(config, W, mention_idx).to(torch_device)
    model2.model = model.model
    model2.lm_head = model.lm_head
    model2.final_logits_bias = model.final_logits_bias
    model2.eval()
    out2 = model2(input_ids, dots, labels = labels)
    out3 = model2(input_ids, dots, labels = labels)

    print(out2.loss)
    print(out3.loss)

    dots_input = W(dots.view(bsz, 7, 4))
    emb = model2.get_input_embeddings()
    enc = model2.model.get_encoder()
    embedding_dim = emb.embedding_dim

    tokens_input = emb(input_ids)
    inputs_embeds = torch.cat([dots_input, tokens_input], 1)
    inputs_embeds = inputs_embeds * enc.embed_scale

    # cant use generate
    #gen2 = model2.generate(inputs_embeds = inputs_embeds)

    out3.loss.backward()
    print(model2.dot_encoder.weight.grad)

    import pdb; pdb.set_trace()

