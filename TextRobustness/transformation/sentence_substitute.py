"""
WordSubstitute Base Class
============================================

"""

from functools import reduce
from abc import abstractmethod
from TextRobustness.transformation import Transformation


class SentenceSubstitute(Transformation):
    """ Word replace transformation to implement normal word replace functions.

    Attributes:
        trans_min: int
            Minimum number of character will be augmented.
        trans_max: int
            Maximum number of character will be augmented. If None is passed, number of augmentation is
            calculated via aup_char_p. If calculated result from aug_p is smaller than aug_max, will use calculated
            result from aup_char_p. Otherwise, using aug_max.
        trans_p: float
            Percentage of character (per token) will be augmented.
        stop_words: list
            List of words which will be skipped from augment operation.
        processor: TextRobustness.common.preprocess.TextProcessor.

    """
    def __init__(self, max_try_times=3, **kwargs):
        super().__init__()
        self.max_try_times = max_try_times

    def __repr__(self):
        return 'Transformation'

    def _transform(self, sample, field='x', n=5, **kwargs):
        """ Transform text string according field.

        Args:
            sample: dict
                input data, normally one data component.
            fields: str or list
                indicate which fields to transform,
                for multi fields , substitute them at the same time.
            n: int
                number of generated samples
        Returns:
            list
                transformed sample list.
        """
        # process multi field transformation
        fields = [field] if isinstance(field, str) else field
        # shape: (fields_num  * n)
        fields_replace_sentences = []

        # Watch out! multi fields transform separately
        for i, field in enumerate(fields):
            tokens = sample.get_words(field)
            for time in range(self.max_try_times):
                rep_sentences = self._get_replace_sentences(tokens, n=n)
                if rep_sentences != [[]]:
                    fields_replace_sentences.append(rep_sentences)
                    break

        if fields_replace_sentences == [[]]:
            return []

        # skip when no enough candidates
        if len(fields_replace_sentences) != len(fields):
            return []

        # align different fields to same shape
        sentences_num = [[len(words) for words in rep_sentences] for rep_sentences in fields_replace_sentences]
        rep_num = min(reduce(lambda x, y: x + y, sentences_num))

        if sentences_num:
            rep_num = min(reduce(lambda x, y: x+y, sentences_num))

        trans_samples = []
        # get substitute candidates combinations

        replace_input = []
        for i in range(rep_num):
            for field_index in range(len(fields)):
                for index in range(len(fields_replace_sentences[field_index])):
                    sentence = fields_replace_sentences[field_index][index][i]
                    replace_input.append(sentence)

        for rep in replace_input:
            # shape: (fields_num * n)
            trans_samples.append(sample.replace_field(field, rep))

        return trans_samples

    @abstractmethod
    def _get_replace_sentences(self, words, n=5):
        """ Returns a list containing all possible words .

        Args:
            words: list of str

        Returns:
            list of list
        """
        raise NotImplementedError
