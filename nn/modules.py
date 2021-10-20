import sys
import os

import torch
from torch import nn
from pytorch_pretrained_bert import BertModel


class BERT(nn.Module):

    HIDDEN_DIM = 768

    def __init__(self, resource_dir):
        super(BERT, self).__init__()

        model_path = os.path.join(resource_dir, "pretrained_lm/model.pt")
        self._bert_model = BertModel.from_pretrained(model_path)

    def forward(self, var_h, mask_mat, var_pos):
        lm_repr, _ = self._bert_model(var_h, attention_mask=mask_mat, output_all_encoded_layers=False)

        batch_size, _, hidden_dim = lm_repr.size()
        _, token_num = var_pos.size()
        exp_pos = var_pos.unsqueeze(-1).expand(batch_size, token_num, hidden_dim)
        return torch.gather(lm_repr, dim=-2, index=exp_pos)


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()

        self._activator = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.Tanh(),
                                        nn.Linear(hidden_dim, output_dim))
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, var_h):
        return self._activator(self._dropout(var_h))
