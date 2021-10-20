import os

from pytorch_pretrained_bert import BertTokenizer


class WordAlphabet(object):
    pass


class LabelAlphabet(object):

    def __init__(self):
        self._idx_to_item = []
        self._item_to_idx = {}

    def add(self, item):
        if item not in self._item_to_idx:
            self._item_to_idx[item] = len(self._idx_to_item)
            self._idx_to_item.append(item)

    def get(self, idx):
        return self._idx_to_item[idx]

    def index(self, item):
        return self._item_to_idx[item]

    def __str__(self):
        return str(self._item_to_idx)

    def __len__(self):
        return len(self._idx_to_item)


class PieceAlphabet(object):

    CLS_SIGN, SEP_SIGN = "[CLS]", "[SEP]"
    PAD_SIGN, SOS_SIGN = "[PAD]", "[SOS]"
    SPACE_SIGN = "[SPACE]"

    def __init__(self, resource_dir):
        dict_path = os.path.join(resource_dir, "pretrained_lm", "vocab.txt")

        # Manually add last two symbols to vocab.txt.
        no_splits = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[SOS]", "[SPACE]"]
        self._segment_model = BertTokenizer(vocab_file=dict_path, never_split=no_splits, do_lower_case=False)

    def tokenize(self, item):
        return self._segment_model.tokenize(item)

    def index_seq(self, items):
        return self._segment_model.convert_tokens_to_ids(items)

    def get_seq(self, indexes):
        return self._segment_model.convert_ids_to_tokens(indexes)
