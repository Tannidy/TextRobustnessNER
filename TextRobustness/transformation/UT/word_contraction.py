"""
contract or extend sentence by common abbreviations
==========================================================
"""
import re
from TextRobustness.component.sample import SASample
from TextRobustness.transformation import Transformation
from TextRobustness.common.settings import CONTRACTION_MAP, REVERSR_CONTRACTION_MAP


class WordContraction(Transformation):
    """
    Transforms input by common abbreviations
    """

    def __init__(self, processor=None):
        super().__init__(processor=processor)

    def _transform(self, sample, transform_field='x', **kwargs):
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
        transform_fields = [transform_field] if isinstance(transform_field, str) else transform_field

        trans_samples = []

        for i, field in enumerate(transform_fields):
            tokens = sample.get_words(field)
            get_contract_sample = self._get_contract_sample(tokens, field, sample)
            if get_contract_sample:
                trans_samples.append(get_contract_sample)
            get_reverse_contract_sample = self._get_reverse_contract_sample(tokens, field, sample)
            if get_reverse_contract_sample:
                trans_samples.append(get_reverse_contract_sample)

            return trans_samples

    @staticmethod
    def _get_contract_sample(tokens, field, sample):
        contract_sample = sample
        indexes_list = []
        contract_list = []
        for i in range(len(tokens)):
            if len(tokens) > i + 1:
                judge_string = tokens[i] + ' ' + tokens[i + 1]
                if judge_string in CONTRACTION_MAP:
                    words = CONTRACTION_MAP[judge_string].split('\'')
                    words[1] = '\'' + words[1]
                    indexes_list.append([i, i + 1])
                    contract_list.append(words)

        if indexes_list:
            if len(indexes_list) == len(contract_list):
                for i, indexes in enumerate(indexes_list):
                    contract_sample = contract_sample.replace_field_at_indices(field, indexes, contract_list[i])

            return contract_sample

    @staticmethod
    def _get_reverse_contract_sample(tokens, field, sample):
        reverse_sample = sample
        indexes_list = []
        reverse_list = []
        for i in range(len(tokens)):
            if len(tokens) > i + 2:
                judge_string = tokens[i] + tokens[i + 1] + tokens[i + 2]
                if judge_string in REVERSR_CONTRACTION_MAP:
                    words = REVERSR_CONTRACTION_MAP[judge_string].split()
                    words.append(' ')
                    indexes_list.append([i, i + 1, i + 2])
                    reverse_list.append(words)

        if indexes_list:
            if len(indexes_list) == len(reverse_list):
                for i, indexes in enumerate(indexes_list):
                    reverse_sample = reverse_sample.replace_field_at_indices(field, indexes, reverse_list[i])

            new_tokens = reverse_sample.get_words(field)
            space_list = []
            for i in range(len(new_tokens)):
                if new_tokens[i] == ' ':
                    space_list.insert(0, i)

            for i in space_list:
                reverse_sample = reverse_sample.delete_field_at_index(field, i)

            return reverse_sample


if __name__ == "__main__":
    sent1 = "we're you are he's"
    data_sample = SASample({'x': sent1, 'y': "negative"})
    swap_ins = WordContraction()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
