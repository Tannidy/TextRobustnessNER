"""
Swapping word synonyms in WordNet
==========================================================
"""

from TextRobustness.transformation import WordSubstitute


class WordNetAntonym(WordSubstitute):
    """ Transforms an input by replacing its words with antonym provided by
        WordNet. Download nltk_data before running.

        Just support adj pos word antonym replace.

    """
    def __init__(self, trans_min=1, trans_max=10, trans_p=0.1,
                 stop_words=None, language="eng", **kwargs):
        super().__init__(trans_min=trans_min, trans_max=trans_max, trans_p=trans_p,
                         stop_words=stop_words)
        self.language = language
        self.get_pos = True

    def __repr__(self):
        return 'WordNetAntonym'

    def _get_candidates(self, word, n=5, pos=None, **kwargs):
        """ Returns a list containing all possible words with 1 character
            replaced by a homoglyph.

        """
        candidates = set()
        # filter different pos in get_wsd function
        antonyms = self.processor.get_antonyms([(word, pos)])[0]

        for antonym in antonyms:
            for ant_word in antonym.lemma_names(lang=self.language):
                if (
                    (ant_word != word)
                    and ("_" not in ant_word)
                ):
                    # WordNet can suggest phrases that are joined by '_' but we ignore phrases.
                    candidates.add(ant_word)

        if not candidates:
            return []

        return list(candidates)[:n]

    def skip_aug(self, tokens, mask, pos=None):
        """ Skip non adj word.

        Args:
            tokens: list
            mask: list
            pos: list

        Returns:
            list of allowed indices.

        """
        nor_pos = []
        adj_indices = []

        for pos_tag in pos:
            nor_pos.append(self.processor.normalize_pos(pos_tag))

        indices = self.pre_skip_aug(tokens, mask)

        for index in indices:
            if nor_pos[index] == 'a':
                adj_indices.append(index)

        return adj_indices


if __name__ == "__main__":
    from TextRobustness.component.sample import SASample
    sent1 = 'The fast brown fox jumps over the lazy dog .'
    data_sample = SASample({'x': sent1, 'y': "negative"})
    swap_ins = WordNetAntonym()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
