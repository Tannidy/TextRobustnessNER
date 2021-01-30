"""
Word Swap by swapping word to double denial forms
==========================================================
"""
from TextRobustness.component.sample import SASample
from TextRobustness.transformation import Transformation
from TextRobustness.common.settings import SA_DOUBLE_DENIAL_DICT


class WordSwapDoubleDenial(Transformation):
    """Transforms an input by replacing its words with double denial forms.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.polarity_dict = SA_DOUBLE_DENIAL_DICT

    def _transform(self, sample, n=5, **kwargs):
        """ Transform text string according field.

        Args:
            sample: dict
                input data, normally one data component.
            n: int
                number of generated samples, but this method can only generate one sample
        Returns:
            list
                transformed sample list.
        """

        tokens = sample.get_value('x')
        swap_indices, swap_tokens = self._get_double_denial_info(tokens)
        if not swap_indices:
            return []

        trans_samples = [sample.replace_fields_at_indices(['x'], [swap_indices], [swap_tokens])]
        return trans_samples

    def _get_double_denial_info(self, tokens):
        """ get words that can be converted to a double denial form

        Args:
            tokens: list
                tokenized words
        Returns:
            indices: list
                indices of tokens that should be replaced
            double_denial_words: list
                The new words that correspond to indices and is used to replace them

        """
        if tokens is str:
            tokens = [tokens]

        indices = []
        double_denial_words = []
        for word in self.polarity_dict.keys():
            word_cnt = tokens.count(word)
            current_index = 0
            for i in range(word_cnt):
                current_index = tokens.index(word, current_index + 1)
                indices.append(current_index)
                double_denial_words.append([self.polarity_dict[word]])
        return indices, double_denial_words


if __name__ == "__main__":
    sent1 = "It's a good movie."
    data_sample = SASample({'x': sent1, 'y': "negative"})
    swap_ins = WordSwapDoubleDenial()
    x = swap_ins.transform(data_sample, n=5)

    for sa_sample in x:
        print(sa_sample.dump())
