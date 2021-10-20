import torch
from torch import nn


class StandardDP(nn.Module):

    _NEG_INFINITY_VAL = -999999.0
    _NONE_TRACE_VAL, _SOS_TRACE_VAL = -2, -1

    def __init__(self):
        super(StandardDP, self).__init__()

    def forward(self, score_table):
        label_table, label_trace = torch.max(score_table, dim=-1)
        batch_size, seq_len, _ = label_table.size()

        span_table = torch.zeros(batch_size, seq_len).float()
        span_table.fill_(StandardDP._NEG_INFINITY_VAL)
        span_trace = torch.zeros(batch_size, seq_len).long()
        span_trace.fill_(StandardDP._NONE_TRACE_VAL)

        if torch.cuda.is_available():
            span_table = span_table.cuda()
            span_trace = span_trace.cuda()
        span_table[:, 0] = label_table[:, 0, 0]
        span_trace[:, 0] = StandardDP._SOS_TRACE_VAL

        for i in range(1, seq_len):
            candidates = span_table[:, :i] + label_table[:, 1: i + 1, i]
            transit_val, transit_idx = torch.max(candidates, dim=-1)

            span_table[:, i] = transit_val
            span_trace[:, i] = transit_idx
        return span_table, label_trace, span_trace
