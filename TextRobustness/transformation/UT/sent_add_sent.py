"""
extend sentences by irrelevant sentences
==========================================================
"""
import random
from TextRobustness.common.settings import SENT_PATH
from TextRobustness.component.sample import SASample
from TextRobustness.transformation import Transformation
from TextRobustness.common.settings import MIN_SENT_TRANS_LENGTH
from TextRobustness.common.utils.load import load_sentences

sent_list = load_sentences(SENT_PATH)


class SentAddSent(Transformation):
    """
    Transforms input by adding sentences
    """

    def __init__(self, processor=None):
        super().__init__(processor=processor)

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
        transform_fields = [transform_field] if isinstance(transform_field, str) else transform_field

        trans_samples = []

        for i, field in enumerate(transform_fields):
            tokens = sample.get_words(field)
            get_samples = self._get_add_sent_sample(tokens, field, sample, n)
            if get_samples:
                for i in get_samples:
                    trans_samples.append(i)

        if trans_samples:
            if len(trans_samples) > n:
                return trans_samples[:n]
            else:
                return trans_samples

    @staticmethod
    def _get_add_location(tokens):
        if len(tokens) < MIN_SENT_TRANS_LENGTH:
            return []
        add_location = []
        for i, word in enumerate(tokens):
            if i == 0:
                add_location.append(i)
            elif word in ['.', ',', '...', '?', '!']:
                add_location.append(i)

        return add_location

    def _get_add_sent_sample(self, tokens, field, sample, n):
        add_location = self._get_add_location(tokens)
        if not add_location:
            return []
        else:
            transformed_samples = self._get_sample(tokens, field, sample, add_location, n)
            return transformed_samples

    def _get_sample(self, tokens, field, sample, add_location, n):
        sample_list = []
        for i in range(n):
            add_sent_sample = sample
            add_loc = random.choice(add_location)
            add_sent = random.choice(sent_list)
            add_tokens = self.processor.word_tokenize(add_sent)
            for i, token in enumerate(add_tokens):
                add_sent_sample = add_sent_sample.insert_field_before_index(field, add_loc + i, token)
            sample_list.append(add_sent_sample)
        return sample_list


if __name__ == "__main__":
    sent1 = 'The quick brown fox, jumps over the lazy dog .'
    data_sample = SASample({'x': sent1, 'y': "negative"})
    swap_ins = SentAddSent()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
