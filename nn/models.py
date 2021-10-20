import torch
from torch import nn

from nn.scorers import BERTCharting
from nn.dp import StandardDP


class StandardLUA(nn.Module):

    def __init__(self,
                 label_num: int,
                 hidden_dim: int,
                 dropout_rate: float,
                 resource_dir: str):
        super(StandardLUA, self).__init__()

        self._scorer_model = BERTCharting(label_num, hidden_dim, dropout_rate, resource_dir)
        self._solver_model = StandardDP()

    def forward(self, var_seq, var_mask, var_pos):
        pass

    def estimate(self, var_seq, var_mask, var_pos, var_len, var_label=None):
        score_table = self._scorer_model(var_seq, var_mask, var_pos)

        dp_table, label_trace, span_trace = self._solver_model(score_table)
        exp_len = var_len.unsqueeze(-1)
        predicted_scores = torch.gather(dp_table, dim=-1, index=exp_len - 1).squeeze()

        masked_score = score_table.masked_fill(var_label, 0.0)
        gold_scores = torch.sum(torch.sum(torch.sum(masked_score, dim=-1), dim=-1), dim=-1)
        return torch.mean(predicted_scores - gold_scores)

    def inference(self, var_seq, var_mask, var_pos, var_len, label_vocab):
        score_table = self._scorer_model(var_seq, var_mask, var_pos)
        _, label_trace, span_trace = self._solver_model(score_table)

        seq_lens = var_len.cpu().numpy().tolist()
        label_trace = label_trace.cpu().numpy().tolist()
        span_trace = span_trace.cpu().numpy().tolist()

        segments = []
        for size, labels, prefixes in zip(seq_lens, label_trace, span_trace):
            segments.append([])
            cursor = size - 1

            while cursor != -1:
                split = prefixes[cursor]
                predicted_l = label_vocab.get(labels[split + 1][cursor])
                segments[-1].append((split + 1, cursor, predicted_l))
                cursor = split
        return [u[::-1] for u in segments]
