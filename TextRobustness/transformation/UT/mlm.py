"""
Swapping words by Mask Language Model
==========================================================
"""
import string

import torch
from collections import OrderedDict
from TextRobustness.component.sample import SASample

from TextRobustness.transformation import WordSubstitute
from TextRobustness.common.settings import BERT_MODEL_NAME


# TODO, 修改至可用
class MLM(WordSubstitute):
    """
    Transforms an input by replacing its tokens with words of mask language predicted.

    Attributes:
        accrue_threshold: threshold of Bert results to pick
    """
    def __init__(self, accrue_threshold=1, trans_min=1, trans_max=10, trans_p=0.1,
                 stop_words=None):
        super().__init__(trans_min=trans_min, trans_max=trans_max, trans_p=trans_p,
                         stop_words=stop_words)
        self.get_pos = True
        self.accrue_threshold = accrue_threshold

        from transformers import BertTokenizer, BertForMaskedLM
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained(BERT_MODEL_NAME)

    def __repr__(self):
        return 'MLM'

    def _get_substitute_words(self, tokens, indexes, n=5):
        """ Returns a list containing all possible words with 1 word replaced by a Bert."""
        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        words_list = []

        for index in indexes:
            index = index + 1
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            segments_ids = [0] * len(tokens)
            tokens[index] = "[MASK]"
            indexed_tokens[index] = 103

            # convert to tensor
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            results_dict = {}

            with torch.no_grad():
                predictions = self.model(tokens_tensor, segments_tensors)
                for i in range(len(predictions[0][0, index])):
                    if float(predictions[0][0, index][i].tolist()) > self.accrue_threshold:
                        replace_word = self.tokenizer.convert_ids_to_tokens([i])[0]
                        results_dict[replace_word] = float(predictions[0][0, index][i].tolist())

            sorted_dict = OrderedDict(sorted(results_dict.items(), key=lambda kv: kv[1], reverse=True))

            bert_words = []
            # TODO POS
            for i, word in enumerate(sorted_dict):
                if word in string.punctuation or word.startswith('##') or len(word) == 1 \
                        or word.startswith('.') or word.startswith('[') or word == tokens[index]:
                    continue
                else:
                    bert_words.append(word)
                if len(bert_words) == n:
                    break
            words_list.append(bert_words)

        return words_list

    def skip_aug(self, tokens, mask):
        return self.pre_skip_aug(tokens, mask)

    def pre_skip_aug(self, tokens, mask):
        results = []
        for token_idx, token in enumerate(tokens):
            # skip punctuation
            if token in string.punctuation:
                continue
            # skip stopwords by list
            if self.is_stop_words(token):
                continue
            if token in ['[CLS]', '[SEP]']:
                continue

            results.append(token_idx)

        return results


if __name__ == "__main__":
    sent1 = 'The quick brown fox jumps over the lazy dog .'
    data_sample = SASample({'x': sent1, 'y': "negative"})
    swap_ins = MLM()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
