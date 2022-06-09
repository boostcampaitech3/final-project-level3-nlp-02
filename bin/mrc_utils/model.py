import os
import sys
import torch
import torch.nn as nn
import numpy as np

from mrc_utils.datasets import DatasetDict, load_from_disk, load_metric
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    ElectraTokenizer,
    ElectraTokenizerFast,
    ElectraModel,
    #ElectraModelForQuestionAnswering,
    ElectraForQuestionAnswering,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,

)
from transformers.modeling_outputs import QuestionAnsweringModelOutput

def get_config_tokenizer_model(model_args):

    config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name is not None else model_args.model_name_or_path)

    if 'electra' in model_args.model_name_or_path.lower():
        tokenizer = ElectraTokenizerFast.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name_or_path, use_fast=True)
        model = ElectraForQuestionAnswering.from_pretrained( model_args.model_name_or_path, from_tf=bool(".ckpt" in model_args.model_name_or_path), config=config)
        # b_model = ElectraModel.from_pretrained( model_args.model_name_or_path, from_tf=bool(".ckpt" in model_args.model_name_or_path), config=config)
        # b_model_output_dim = 768 # b_model.pooler.dense.out_features
        # model = TunedElectra(b_model, b_model_output_dim)

    else:
        tokenizer = AutoTokenizer.from_pretrained( model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name_or_path, use_fast=True)
        model = AutoModelForQuestionAnswering.from_pretrained( model_args.model_name_or_path, from_tf=bool(".ckpt" in model_args.model_name_or_path), config=config)

    return config, tokenizer, model


class TunedElectra(nn.Module):
    def __init__(self, base_model, b_model_output_dim):
        super().__init__()
        self.base_model = base_model
        self.b_model_output_dim = b_model_output_dim

        self.linear = nn.Linear(self.b_model_output_dim, 2)

    def forward(self, input_ids, attention_mask,*args,**kwargs):
        base_outputs = self.base_model(input_ids, attention_mask,*args,**kwargs)
        linear_outputs = self.linear(base_outputs.last_hidden_state)
        loss = None
        start_logits = linear_outputs[:,:,0]
        end_logits = linear_outputs[:,:,1]
        qamo = QuestionAnsweringModelOutput(start_logits=start_logits, end_logits = end_logits)
        print('qamo',type(qamo))
        print('logits_shape', start_logits.shape, end_logits.shape)
        return qamo

