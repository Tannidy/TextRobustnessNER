"""
Add word with given pos tags from BERT before word with given pos tags
==========================================================
"""

from TextRobustness.component.sample import POSSample
from TextRobustness.transformation.POS import MultiPOSWordSwapWordNet
from transformers import BertTokenizer, BertForMaskedLM
import copy
from TextRobustness.common.settings import BERT_MODEL_NAME
from TextRobustness.common.utils.mlm_predictions import prediction_from_model


class SwapWordBERT(MultiPOSWordSwapWordNet):
    """Transforms an input by replacing its words with synonyms provided by
    BERT. Download nltk_data before running."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained(BERT_MODEL_NAME)
        self.model.eval()

    def sample_words(self, tokens, index, candidates, n):
        """See POSBaseTrans.
        """
        _tokens = copy.deepcopy(tokens)
        _tokens[index] = self.tokenizer.mask_token
        _tokens = [self.tokenizer.cls_token] + _tokens + [self.tokenizer.sep_token]

        replace_words = prediction_from_model(self.model, self.tokenizer, _tokens)[0]

        filtered_results = []
        for word in replace_words:
            if len(filtered_results) >= n:
                break
            if word in self.stop_words or word not in candidates:
                continue
            filtered_results.append(word)
        return filtered_results


if __name__ == "__main__":
    x = "That is his survey".split()
    y = "DT VBZ DT NN".split()

    data_sample = POSSample({'x': x, 'y': y})
    add_ins = SwapWordBERT(treebank_tags=['NN'])
    x = add_ins.transform(sample=data_sample, field=['x', 'x_mask'], n=3)

    for sample in x:
        print(sample.dump())
