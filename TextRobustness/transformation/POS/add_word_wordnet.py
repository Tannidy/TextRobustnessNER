"""
Add word with given pos tags from wordnet before word with given pos tags
==========================================================
"""

from TextRobustness.component.sample import POSSample
from TextRobustness.transformation.POS import POSBaseTrans
import string


class AddWordWordNet(POSBaseTrans):
    """Transforms an input by adding words with specific POS tags based on
    WordNet. Download nltk_data before running.

    Args:
        treebank_tags: insert before the words with thess pos tags
        add_treebank_tags: the pos tags for the inserted words
        language: default english
        **kwargs: same as WordSubstitute
    """

    def __init__(self, treebank_tags, add_treebank_tags, **kwargs):
        super().__init__(**kwargs)
        support_pos = ["JJ", "NN", "VB", "RB"]
        for treebank_tag in treebank_tags:
            assert treebank_tag in support_pos, "Only support replacing JJ, NN, VB and RB!"
        for treebank_tag in add_treebank_tags:
            assert treebank_tag in support_pos, "Only support replacing JJ, NN, VB and RB!"

        self.treebank_tags = treebank_tags
        self.add_treebank_tags_dict = {i: j for i, j in zip(treebank_tags, add_treebank_tags)}
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
            "NN": list(noun),
            "VB": list(verb),
            "JJ": list(adj),
            "RB": list(adv)
        }
        return candidates_dict

    def _get_substitute_words(self, field, sample, indices, n=5):
        """See POSBaseTrans.
        """
        assert field == ['x', 'y', 'x_mask']
        trans_words = []
        trans_labels = []
        trans_masks = []
        for index in indices:
            token = sample.get_value('x')[index]
            label = sample.get_value('y')[index]
            trans_label = self.add_treebank_tags_dict[label]
            trans_word_list = self.sample_words(sample.get_value('x'), index, self.candidates_dict[trans_label], n)

            trans_words.append([[trans_word, token] for trans_word in trans_word_list])
            trans_labels.append([[trans_label, label]] * len(trans_word_list))
            trans_masks.append([[0, 1]] * len(trans_word_list))
        return [trans_words, trans_labels, trans_masks]

    def skip_aug(self, tokens, labels):
        """See POSBaseTrans.
        """
        results = []
        for token_idx, (token, label) in enumerate(zip(tokens, labels)):
            # skip first word
            if token_idx == 0:
                continue
            # skip non-chosen tag
            if label not in self.treebank_tags:
                continue
            # only add before whole phrase (e.g., NNP)
            if token_idx > 0 and labels[token_idx - 1] == label:
                continue
            # to keep fluency, e.g., ignore the NN that already have JJ before
            if token_idx > 0 and labels[token_idx - 1] == self.add_treebank_tags_dict[label]:
                continue
            # skip punctuation
            if token in string.punctuation:
                continue
            # skip stopwords by list
            if self.is_stop_words(token):
                continue

            results.append(token_idx)

        return results


if __name__ == "__main__":
    x = "That is a survey".split()
    y = "DT VBZ DT NN".split()

    data_sample = POSSample({'x': x, 'y': y})
    add_ins = AddWordWordNet(['NN'], ['JJ'])
    x = add_ins.transform(sample=data_sample, field=['x', 'y', 'x_mask'], n=3)

    for sample in x:
        print(sample.dump())
