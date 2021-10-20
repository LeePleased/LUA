import json
import os
import time
from tqdm import tqdm
import argparse

import torch
from pytorch_pretrained_bert import BertAdam

from utils import fix_random_seed
from utils import get_corpus_iterator
from nn import StandardLUA
from utils import data_to_tensor
from utils import Procedure
from utils import warmup_linear
from utils import PieceAlphabet
from utils import LabelAlphabet


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", "-dd", type=str, required=True)
parser.add_argument("--save_dir", "-sd", type=str, required=True)
parser.add_argument("--resource_dir", "-rd", type=str, required=True)
parser.add_argument("--random_state", "-rs", type=int, default=0)
parser.add_argument("--epoch_num", "-en", type=int, default=15)
parser.add_argument("--batch_size", "-bs", type=int, default=8)

parser.add_argument("--hidden_dim", "-hd", type=int, default=256)
parser.add_argument("--dropout_rate", "-dr", type=float, default=0.3)

args = parser.parse_args()
print(json.dumps(args.__dict__, ensure_ascii=False, indent=True), end="\n\n")

fix_random_seed(args.random_state)

token_vocab, label_vocab = PieceAlphabet(args.resource_dir), LabelAlphabet()
train_loader = get_corpus_iterator(os.path.join(args.data_dir, "train.json"), args.batch_size, True, label_vocab)
dev_loader = get_corpus_iterator(os.path.join(args.data_dir, "dev.json"), args.batch_size, False)
test_loader = get_corpus_iterator(os.path.join(args.data_dir, "test.json"), args.batch_size, False)

model = StandardLUA(len(label_vocab), args.hidden_dim, args.dropout_rate, args.resource_dir)
model = model.cuda() if torch.cuda.is_available() else model.cpu()

all_parameters = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
LEARNING_RATE, WEIGHT_DECAY = (5e-5, 1e-5)  # Another good pair: (1e-5, 0.01).
WARMUP_PROPORTION = 0.1
grouped_param = [{'params': [p for n, p in all_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
                 {'params': [p for n, p in all_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
total_steps = int(len(train_loader) * (args.epoch_num + 1))
optimizer = BertAdam(grouped_param, lr=LEARNING_RATE, warmup=WARMUP_PROPORTION, t_total=total_steps)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
torch.save(token_vocab, os.path.join(args.save_dir, "word_vocab.pt"))
torch.save(label_vocab, os.path.join(args.save_dir, "label_vocab.pt"))

best_dev_f1, current_step = 0.0, 0
for epoch_idx in range(0, args.epoch_num + 1):
    model.train()
    time_start, train_loss = time.time(), 0.0

    for batch in tqdm(train_loader, ncols=70):
        penalty = model.estimate(*data_to_tensor(batch, token_vocab, label_vocab, True))
        train_loss += penalty.cpu().item() * len(batch)

        present_lr = LEARNING_RATE * warmup_linear(1.0 * current_step / total_steps, WARMUP_PROPORTION)
        for params_group in optimizer.param_groups:
            params_group['lr'] = present_lr
        current_step += 1

        optimizer.zero_grad()
        penalty.backward()
        optimizer.step()

    train_time = time.time() - time_start
    print("(Epoch {:5d}) training loss: {:.6f}, training time: {:.3f}".format(epoch_idx, train_loss, train_time))

    dev_score, dev_time = Procedure.evaluation(model, dev_loader, token_vocab, label_vocab, args.resource_dir)
    print("(Epoch {:5d}) dev score: {:.6f}, dev time: {:.3f}".format(epoch_idx, dev_score, dev_time))
    test_score, test_time = Procedure.evaluation(model, test_loader, token_vocab, label_vocab, args.resource_dir)
    print("[Epoch {:5d}] test score: {:.6f}, test time: {:.3f}".format(epoch_idx, test_score, test_time))

    if dev_score >= best_dev_f1:
        best_dev_f1 = dev_score

        print("\n<Epoch {:5d}> best model, test score: {:.6f}".format(epoch_idx, test_score))
        torch.save(model, os.path.join(args.save_dir, "model.pt"))
    print(end="\n\n")
