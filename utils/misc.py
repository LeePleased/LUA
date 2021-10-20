import random
import os
import codecs
import json
import time
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.eval import CoNLL_f1


def iterative_support(func, query):
    if isinstance(query, (list, tuple, set)):
        return [iterative_support(func, i) for i in query]
    return func(query)


def warmup_linear(x, warmup=0.002):
    """This is copied from pytorch_pretrained_bert.optimization (version 0.4.0)."""

    if x < warmup:
        return x / warmup
    return 1.0 - x


def fix_random_seed(state_val):
    random.seed(state_val)
    np.random.seed(state_val)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(state_val)
        torch.cuda.manual_seed_all(state_val)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(state_val)
    torch.random.manual_seed(state_val)


def load_data_from_json(file_path):
    with codecs.open(file_path, "r", "utf-8") as fr:
        data = json.load(fr)
    return data


def dump_data_to_json(data, file_path):
    with codecs.open(file_path, "w", "utf-8") as fw:
        json.dump(data, fw, ensure_ascii=False, indent=True)


class FormatConversion(object):

    @staticmethod
    def iob_to_segments(labels):
        seq_len, chunks, cursor = len(labels), [], 0

        while cursor < seq_len:
            tag = labels[cursor]
            pointer = cursor + 1

            if tag == "O":
                chunks.append((cursor, pointer - 1, "O"))
            else:
                group, label = tag.split("-")
                assert group in ["B", "I"]

                while pointer < seq_len:
                    nxt_t = labels[pointer]
                    if nxt_t == "O":
                        break
                    else:
                        nxt_g, nxt_l = nxt_t.split("-")
                        if nxt_g == "B":
                            assert nxt_l == label
                            break
                        elif nxt_l != label:
                            assert nxt_g == "I"
                            break
                        else:
                            assert nxt_g == "I" and nxt_l == label
                    pointer += 1

                chunks.append((cursor, pointer - 1, label))
            cursor = pointer
        return chunks

    @staticmethod
    def iob2_to_segments(label_list):
        seq_len, chunks = len(label_list), []

        cursor = 0
        while cursor < seq_len:
            tag = label_list[cursor]
            pointer = cursor + 1

            if tag == "O":
                chunks.append((cursor, pointer - 1, "O"))
            else:
                group, label = tag.split("-")
                assert group == "B"

                while pointer < seq_len:
                    nxt_t = label_list[pointer]
                    if nxt_t == "O":
                        break
                    else:
                        nxt_g, nxt_l = nxt_t.split("-")
                        if nxt_g == "B":
                            break
                        else:
                            assert nxt_g == "I"
                            assert nxt_l == label
                    pointer += 1

                chunks.append((cursor, pointer - 1, label))
            cursor = pointer
        return chunks

    @staticmethod
    def segments_to_iob2(atoms):
        tags = []

        for left_b, right_b, group in atoms:
            num = right_b - left_b + 1
            if group == "O":
                tags.extend(["O"] * num)
            else:
                tags.extend(["B-" + group] + ["I-" + group] * (num - 1))
        return tags


def get_corpus_iterator(file_path, batch_size, is_shuffle, label_vocab=None):
    material = load_data_from_json(file_path)
    data = [(eval(elem["text"]), eval(elem["segments"])) for elem in material]
    if label_vocab is not None:
        iterative_support(label_vocab.add, [[l for _, _, l in chunks] for _, chunks in data])

    class _DataSet(Dataset):

        def __init__(self, instances):
            self._instances = instances

        def __getitem__(self, item):
            return self._instances[item]

        def __len__(self):
            return len(self._instances)

    data_set = _DataSet(data)
    return DataLoader(data_set, batch_size, is_shuffle, collate_fn=lambda x: x)


def data_to_tensor(batch_data, token_vocab, label_vocab, is_train):
    undone_text, serial_lens, undone_poses, bert_lens = [], [], [], []

    for item in batch_data:
        sent = [token_vocab.SOS_SIGN] + item[0]
        pieces = iterative_support(token_vocab.tokenize, sent)

        units, positions, cursor = [], [], 0
        for tokens in pieces:
            units.extend(tokens)
            positions.append(cursor)
            cursor += len(tokens)

        undone_text.append(units)
        serial_lens.append(len(sent))
        undone_poses.append(positions)
        bert_lens.append(len(units))

    serial_text, serial_masks, serial_poses, serial_labels = [], [], [], []
    piece_num, token_num, case_num = max(bert_lens), max(serial_lens), len(batch_data)

    for i in range(0, case_num):
        assert piece_num >= len(undone_text[i])
        padded_t = [token_vocab.CLS_SIGN] + undone_text[i] + [token_vocab.SEP_SIGN] + (piece_num - len(undone_text[i])) * [token_vocab.PAD_SIGN]
        serial_text.append(token_vocab.index_seq(padded_t))
        serial_masks.append([0 if t == token_vocab.PAD_SIGN else 1 for t in padded_t])
        assert token_num >= len(undone_poses[i])
        serial_poses.append([s + 1 for s in undone_poses[i]] + [piece_num + 1] * (token_num - len(undone_poses[i])))

        if is_train:
            annotation = [(0, 0, "O")] + [(i + 1, j + 1, l) for i, j, l in batch_data[i][1]]
            sign_mat = [[[1] * len(label_vocab) for _ in range(token_num)] for _ in range(token_num)]

            for j, k, l in annotation:
                t = label_vocab.index(l)
                sign_mat[j][k][t] = 0
            serial_labels.append(sign_mat)

    var_text = torch.LongTensor(serial_text)
    var_mask = torch.LongTensor(serial_masks)
    var_len = torch.LongTensor(serial_lens)
    var_position = torch.LongTensor(serial_poses)

    if torch.cuda.is_available():
        var_text = var_text.cuda()
        var_mask = var_mask.cuda()
        var_len = var_len.cuda()
        var_position = var_position.cuda()

    if is_train:
        var_label = torch.LongTensor(serial_labels)

        if torch.cuda.is_available():
            var_label = var_label.cuda()
        return var_text, var_mask, var_position, var_len, var_label == 1
    else:
        return var_text, var_mask, var_position, var_len


class Procedure(object):

    @staticmethod
    def training(model, data_iter, optimizer, token_vocab, label_vocab):
        model.train()
        time_start, total_loss = time.time(), 0.0

        for batch in tqdm(data_iter, ncols=70):
            penalty = model.estimate(*data_to_tensor(batch, token_vocab, label_vocab, True))
            total_loss += penalty.cpu().item() * len(batch)

            optimizer.zero_grad()   # Cleaning up the gradients.
            penalty.backward()      # Back propagation.
            optimizer.step()        # Updating the parameters.

        time_con = time.time() - time_start
        return total_loss, time_con

    @staticmethod
    def evaluation(model, data_iter, token_vocab, label_vocab, source_dir):
        model.eval()
        time_start, text_list = time.time(), []
        predicted_tags, oracle_tags = [], []

        for batch in tqdm(data_iter, ncols=70):
            var_text, var_mask, var_position, var_len = data_to_tensor(batch, token_vocab, label_vocab, False)
            with torch.no_grad():
                segments = model.inference(var_text, var_mask, var_position, var_len, label_vocab)

            text_list.extend([case[0] for case in batch])
            predicted_tags.extend([FormatConversion.segments_to_iob2(case)[1:] for case in segments])
            oracle_tags.extend([FormatConversion.segments_to_iob2(case[1]) for case in batch])

        script_path = os.path.join(source_dir, "conlleval.pl")
        f1_score = CoNLL_f1(text_list, predicted_tags, oracle_tags, script_path)
        return f1_score, time.time() - time_start
