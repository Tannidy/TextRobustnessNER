"""
Word Swap by swaping words that have multiple POS tags in WordNet
==========================================================
"""

from TextRobustness.component.sample import POSSample
from TextRobustness.transformation.POS import POSBaseTrans
import string


class MultiPOSWordSwapWordNet(POSBaseTrans):
    """Word Swap by swaping words that have multiple POS tags in WordNet.
    Download nltk_data before running.

    Args:
        treebank_tags: words with this pos tag will be replaced
        language: default english
        **kwargs: same as WordSubstitute
    """

    def __init__(self, treebank_tags, **kwargs):
        super().__init__(**kwargs)
        support_pos = ["JJ", "NN", "VB", "RB"]
        for treebank_tag in treebank_tags:
            assert treebank_tag in support_pos, "Only support replacing JJ, NN, VB and RB!"

        self.treebank_tags = treebank_tags
        self.candidates_dict = self.get_candidates_dict()

    def get_candidates_dict(self):
        """Get all possible candidates from WordNet.

        Returns:
            dict
        """
        noun = set([i for i in self.processor.get_all_lemmas(pos='n') if "_" not in i])
        verb = set([i for i in self.processor.get_all_lemmas(pos='v') if "_" not in i])
        adj = set([i for i in self.processor.get_all_lemmas(pos='a') if "_" not in i])
        adv = set([i for i in self.processor.get_all_lemmas(pos='r') if "_" not in i])

        candidates_dict = {
            "NN": list(noun & (verb | adj | adv)),
            "VB": list(verb & (noun | adj | adv)),
            "JJ": list(adj & (verb | noun | adv)),
            "RB": list(adv & (verb | adj | noun))
        }
        return candidates_dict

    def _get_substitute_words(self, field, sample, indices, n=5):
        """See POSBaseTrans.
        """
        assert field == ['x', 'x_mask']
        trans_words = []
        trans_masks = []
        for index in indices:
            label = sample.get_value('y')[index]
            trans_word_list = self.sample_words(sample.get_value('x'), index, self.candidates_dict[label], n)

            trans_words.append(trans_word_list)
            trans_masks.append([0] * len(trans_word_list))
        return [trans_words, trans_masks]

    def skip_aug(self, tokens, labels):
        """See POSBaseTrans.
        """
        results = []
        for token_idx, (token, label) in enumerate(zip(tokens, labels)):
            # skip punctuation
            if token in string.punctuation:
                continue
            # skip stopwords by list
            if self.is_stop_words(token):
                continue
            # skip non-chosen tag
            if label not in self.treebank_tags:
                continue

            results.append(token_idx)

        return results


if __name__ == "__main__":
    x = "That is a good survey".split()
    y = "DT VBZ DT JJ NN".split()

    data_sample = POSSample({'x': x, 'y': y})
    swap_ins = MultiPOSWordSwapWordNet(['JJ', "NN"], trans_p=1)
    x = swap_ins.transform(sample=data_sample, field=['x', 'x_mask'], n=3)

    for sample in x:
        print(sample.dump())
