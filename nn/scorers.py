import torch
from torch import nn

from nn.modules import BERT
from nn.modules import MLP


class BERTCharting(nn.Module):

    def __init__(self, label_num, hidden_dim, dropout_rate, resource_dir):
        super(BERTCharting, self).__init__()

        self._encoder = BERT(resource_dir)
        self._scoring_model = MLP(BERT.HIDDEN_DIM * 2, hidden_dim, label_num, dropout_rate)

    def forward(self, var_serial, var_mask, var_pos):
        repr_w = self._encoder(var_serial, var_mask, var_pos)

        batch_size, word_num, hidden_dim = repr_w.size()
        row_h = repr_w.unsqueeze(1).expand(batch_size, word_num, word_num, hidden_dim)
        column_h = repr_w.unsqueeze(2).expand_as(row_h)

        table_h = torch.cat([row_h, column_h], dim=-1)
        return self._scoring_model(table_h)
