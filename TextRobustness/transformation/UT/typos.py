"""
Typos Transformation for add/remove punctuation.
==========================================================

"""

from TextRobustness.common.utils.word_op import *
from TextRobustness.transformation import WordSubstitute


class Typos(WordSubstitute):
    """ Transformation that simulate typos error to transform sentence.

        https://arxiv.org/pdf/1711.02173.pdf

    Attributes:
        trans_min: int
            Minimum number of character will be augmented.
        trans_max: int
            Maximum number of character will be augmented. If None is passed, number of augmentation is
            calculated via aup_char_p. If calculated result from aug_p is smaller than aug_max, will use calculated
            result from aup_char_p. Otherwise, using aug_max.
        trans_p: float
            Percentage of character (per token) will be augmented.
        max_retry_times: int
            try multi times to generate data.
        stop_words: list
            List of words which will be skipped from augment operation.
        processor: TextRobustness.common.preprocess.TextProcessor.
        mode : str.
            just support ['random', 'replace', 'swap', 'insert', 'delete'].
        skip_first_char: bool.
            whether skip the first char of target word.
        skip_last_char: bool.
            whether skip the last char of target word.

    """
    def __init__(self, trans_min=1, trans_max=10, trans_p=0.3, stop_words=None,
                 mode="random", skip_first_char=True, skip_last_char=True, **kwargs):
        super().__init__(trans_min=trans_min, trans_max=trans_max, trans_p=trans_p,
                         stop_words=stop_words)
        self._mode = mode
        self.skip_first_char = skip_first_char
        self.skip_last_char = skip_last_char

    def __repr__(self):
        return 'Typos'

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode_value):
        assert mode_value in ['random', 'replace', 'swap', 'insert', 'delete']
        self._mode = mode_value

    def skip_aug(self, tokens, mask, **kwargs):
        return self.pre_skip_aug(tokens, mask)

    def _get_candidates(self, word, n=5, **kwargs):
        """ Returns a list of words with typo errors.

        Args:
            word: str
            n: int

        Returns:
            list
        """

        candidates = set()

        for i in range(n):
            typo_method = self._get_typo_method()
            # default operate at most one character in a word
            candidates.add(typo_method(word, 1, self.skip_first_char, self.skip_last_char))

        return list(candidates)

    def _get_typo_method(self):
        if self._mode == 'replace':
            return replace
        elif self._mode == 'swap':
            return swap
        elif self._mode == 'insert':
            return insert
        elif self._mode == 'delete':
            return delete
        else:
            return random.choice([replace, swap, insert, delete])


if __name__ == "__main__":
    from TextRobustness.component.sample import SASample

    sample = {'x': 'Pride and Prejudice is a famous fiction', 'y': 'positive'}
    data_sample = SASample(sample)
    typos_trans = Typos(mode='random')

    def test_random():
        typos_trans.mode = 'random'
        print(typos_trans.mode)
        print(data_sample.dump())
        for sample in typos_trans.transform(data_sample, n=3):
            print(sample.dump())

    def test_rep():
        typos_trans.mode = 'replace'
        print(typos_trans.mode)
        print(data_sample.dump())
        for sample in typos_trans.transform(data_sample, n=3):
            print(sample.dump())

    def test_insert():
        typos_trans.mode = 'insert'
        print(typos_trans.mode)
        print(data_sample.dump())
        for sample in typos_trans.transform(data_sample, n=3):
            print(sample.dump())

    def test_delete():
        typos_trans.mode = 'delete'
        print(typos_trans.mode)
        print(data_sample.dump())
        for sample in typos_trans.transform(data_sample, n=3):
            print(sample.dump())

    def test_swap():
        typos_trans.mode = 'swap'
        print(typos_trans.mode)
        print(data_sample.dump())
        for sample in typos_trans.transform(data_sample, n=1):
            print(sample.dump())

    test_random()
    test_rep()
    test_insert()
    test_delete()
    test_swap()
