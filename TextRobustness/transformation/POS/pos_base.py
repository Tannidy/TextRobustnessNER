"""
A base class for POS tagging
==========================================================
"""
from TextRobustness.transformation import WordSubstitute
from abc import abstractmethod


class POSBaseTrans(WordSubstitute):
    """A base class for POS tagging

    """
    def __init__(self, **kwargs):
        super().__init__()

    def _transform(self, sample, field='x', n=5):
        """ Transform text string according field.

        Args:
            sample: dict
                input data, normally one data component.
            field: str or list
                indicate which fields to transform,
                for multi fields , substitute them at the same time.
            n: int
                number of generated samples
        Returns:
            list
                transformed sample list.
        """
        fields = [field] if isinstance(field, str) else field

        # shape: (trans_indices_num)
        trans_indices = self._get_substitute_indices(fields, sample)
        if len(trans_indices) == 0:
            return []

        # shape: (fields_num * trans_indices_num * n * 2) for insertion
        # shape: (fields_num * trans_indices_num * n) for replacement
        trans_items = self._get_substitute_words(fields, sample, trans_indices, n=n)

        # get candidates combinations, contains n rep_input:
        # for insertion: (fields_num * trans_indices_num * 2)
        # for replacement: (fields_num * trans_indices_num)
        rep_input_list = []
        for i in range(n):
            rep_input = []
            for field_index in range(len(fields)):
                rep_input.append([trans_items[field_index][index][i]
                                  for index in range(len(trans_items[field_index]))])
            rep_input_list.append(rep_input)

        # shape: (fields_num * trans_indices_num)
        trans_indices = [trans_indices] * len(fields)

        trans_samples = []
        for rep_input in rep_input_list:
            trans_samples.append(sample.replace_fields_at_indices(fields, trans_indices, rep_input))
        return trans_samples

    def _get_substitute_indices(self, fields, sample):
        """ Returns the index of the replaced tokens.

        Args:
            fields: list or str, the fields to be transform
            sample: POSSample

        Returns:
            list (fields_num * trans_indices_num)
        """
        trans_cnt = self.get_trans_cnt(len(sample.x))
        word_indices = self.skip_aug(sample.x, sample.y)

        if len(word_indices) == 0:
            return []

        if len(word_indices) < trans_cnt:
            trans_cnt = len(word_indices)

        trans_indices = self.sample_num(word_indices, trans_cnt)

        return trans_indices

    @abstractmethod
    def _get_substitute_words(self, field, sample, indices, n=5):
        """Get transform details for each field.

        Args:
            field: list or str, the fields to be transform
            sample: POSSample
            indices: list (fields_num * trans_indices_num)
            n: int

        Returns:
            shape: (fields_num * trans_indices_num * n * 2) for insertion
            shape: (fields_num * trans_indices_num * n) for replacement
        """
        raise NotImplementedError

    @abstractmethod
    def skip_aug(self, tokens, labels):
        """Return the indices that can be transformed.

        Args:
            tokens: list of tokens
            labels: list of labels

        Returns:
            list of int
        """
        raise NotImplementedError

    def sample_words(self, tokens, index, candidates, n):
        """Return n sampled words for word at tokens[index] based on candidates.

        Args:
            tokens: list of tokens
            index: int, the index of word to be operated (replace or insert before)
            candidates: list of tokens
            n: int, the number of words to be returned

        Returns:
            list of tokens
        """
        return self.sample_num(candidates, n)
