"""
Swapping numbers in sentences
==========================================================
"""

import numpy as np
from TextRobustness.component.sample import SASample
from TextRobustness.transformation import SentenceSubstitute


# TODO
class Number(SentenceSubstitute):
    """
    Transforms an input by replacing its numbers.
    """
    def __init__(self, processor=None):
        super().__init__(processor=processor)

    def __repr__(self):
        return 'Number'

    def _get_replace_sentences(self, tokens, n=5, **kwargs):
        replace_sentences = []
        transform_texts_list = self._get_transformation(tokens, n=n)
        if transform_texts_list:
            replace_sentences.append(transform_texts_list)

        return replace_sentences

    def _get_transformation(self, tokens, n):
        digit_location = []
        for i, word in enumerate(tokens):
            if word.isdigit():
                digit_location.append(i)

        if digit_location:
            transformed_texts_list = []
            for i in range(n):
                transformed_texts = self._swap_number(tokens, digit_location)
                transformed_texts_list.append(transformed_texts)
            return transformed_texts_list
        else:
            return []

    @staticmethod
    def _swap_number(tokens, digit_location):
        for i in digit_location:
            number = int(tokens[i])

            if number == 0:
                tokens[i] = str(np.random.randint(1, 100))

            while True:
                if int(tokens[i]) == number:
                    high = number * 5
                    tokens[i] = str(np.random.randint(0, high))
                else:
                    break
        transformed_text = ' '.join(x for x in tokens)

        return transformed_text


if __name__ == "__main__":
    sent1 = 'i am 23 years old .'
    data_sample = SASample({'x': sent1, 'y': "negative"})
    swap_ins = Number()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
