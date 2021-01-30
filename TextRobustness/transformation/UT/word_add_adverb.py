"""
Add adverb word before verb word with given pos tags
==========================================================
"""

import random
from nltk.tag import pos_tag
from TextRobustness.component.sample import SASample
from TextRobustness.common.settings import ADVERB_PATH
from TextRobustness.transformation import WordInsertTransformation
from TextRobustness.common.utils.load import load_adverb_words

adverb_list = load_adverb_words(ADVERB_PATH)


class WordAddAdverb(WordInsertTransformation):
    """
    Transforms an input by add adverb word before verb.
    """

    def __init__(self, processor=None):
        super().__init__(processor=processor)

    def _get_insert_location(self, tokens):
        sentence = ' '.join(word for word in tokens)
        tag = self.processor.get_pos(sentence)
        # tag = pos_tag(tokens)
        verb_location = []
        for i, word_pos in enumerate(tag):
            if word_pos[1] in ['VB', 'VBP', 'VBZ', 'VBG', 'VBD', 'VBN']:
                verb_location.append(i)

        return verb_location

    def _get_insert_word(self, n):
        adverb_words = random.sample(adverb_list, n)
        return adverb_words


if __name__ == "__main__":
    sent1 = 'The quick brown fox jumps over the lazy dog .'
    data_sample = SASample({'x': sent1, 'y': "negative"})
    swap_ins = WordAddAdverb()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
