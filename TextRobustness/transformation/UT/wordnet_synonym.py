"""
Swapping word synonyms in WordNet
==========================================================
"""

from TextRobustness.transformation import WordSubstitute


class WordNetSynonym(WordSubstitute):
    """ Transforms an input by replacing its words with synonyms provided by
        WordNet. Download nltk_data before running.

    """
    def __init__(self, trans_min=1, trans_max=10, trans_p=0.1,
                 stop_words=None, language="eng"):
        super().__init__(trans_min=trans_min, trans_max=trans_max, trans_p=trans_p,
                         stop_words=stop_words)
        self.language = language
        self.get_pos = True

    def __repr__(self):
        return 'WordNetSynonym'

    def _get_candidates(self, word, pos=None, n=5):
        """ Returns a list containing all possible words with 1 character
            replaced by a homoglyph.

        """
        synonym = set()
        # filter different pos in get_wsd function
        synsets = self.processor.get_synsets([(word, pos)])[0]

        for syn in synsets:
            for syn_word in syn.lemma_names(lang=self.language):
                if (
                    (syn_word != word)
                    and ("_" not in syn_word)
                ):
                    # WordNet can suggest phrases that are joined by '_' but we ignore phrases.
                    synonym.add(syn_word)
        if not synonym:
            return []

        return list(synonym)[:n]

    def skip_aug(self, tokens, mask, pos=None):
        return self.pre_skip_aug(tokens, mask)


if __name__ == "__main__":
    from TextRobustness.component.sample import SASample
    sent1 = 'The quick brown fox jumps over the lazy dog .'
    data_sample = SASample({'x': sent1, 'y': "negative"})
    swap_ins = WordNetSynonym()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
