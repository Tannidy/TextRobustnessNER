"""
WordSubstituteTransformation Base Class
============================================

"""
import random
from abc import abstractmethod
from TextRobustness.transformation import Transformation


class WordInsertTransformation(Transformation):
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

    def __init__(self, max_try_times=3, processor=None, **kwargs):
        super().__init__(processor=processor)
        self.max_try_times = max_try_times

    def _transform(self, sample, transform_field='x', n=5, **kwargs):
        """ Transform text string according transform_field.

        Args:
            sample: dict
                input data, normally one data component.
            transform_fields: str or list
                indicate which fields to transform,
                for multi fields , substitute them at the same time.
            n: int
                number of generated samples
        Returns:
            list
                transformed sample list.
        """
        # process multi field transformation
        transform_fields = [transform_field] if isinstance(transform_field, str) else transform_field
        # shape: (fields_num  * n)
        fields_insert_words = []
        fields_insert_indexes = []

        # Watch out! multi fields transform separately
        for i, field in enumerate(transform_fields):
            tokens = sample.get_words(field)
            insert_indexes = self._get_insert_location(tokens)
            if not insert_indexes:
                return []
            for time in range(self.max_try_times):
                select_index = random.choice(insert_indexes)
                insert_words = self._get_insert_word(n=n)
                if insert_words:
                    fields_insert_indexes.append(select_index)
                    fields_insert_words.append(insert_words)
                    break

        if fields_insert_words == [[]]:
            return []

        # skip when no enough candidates
        if len(fields_insert_words) != len(transform_fields):
            return []

        # align different fields to same shape
        insert_word_num = [len(ins_words) for ins_words in fields_insert_words]
        insert_num = min(insert_word_num)

        if insert_word_num:
            insert_num = min(insert_word_num)

        trans_samples = []
        # get substitute candidates combinations

        replace_input = []

        for i, field in enumerate(transform_fields):
            for j in range(insert_num):
                replace_input = fields_insert_words[i][j]
                trans_samples.append(sample.insert_field_before_index(field, fields_insert_indexes[i], replace_input))

        return trans_samples

    @abstractmethod
    def _get_insert_location(self, tokens):
        """ Returns a list containing all possible location to insert .

        Args:
            tokens: list of str

        Returns:
            list
        """
        raise NotImplementedError

    @abstractmethod
    def _get_insert_word(self, n):
        """ Returns a list containing all possible words to insert .

        Args:
            tokens: list of str

        Returns:
            list of word
        """
        raise NotImplementedError
