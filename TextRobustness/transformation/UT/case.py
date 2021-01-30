"""
Word case transformation class
==========================================================
"""

import random

from TextRobustness.transformation import Transformation


class Case(Transformation):
    """ Transforms an input to upper and lower case or capitalize case.

    A sentence can only be transformed into one sentence of each case at most.

    """
    def __init__(self, case_type='upper', **kwargs):
        super().__init__()
        if case_type not in ['upper', 'lower', 'title', 'random']:
            raise ValueError('Not support {0} type, plz ensure case_type in {1}'
                             .format(case_type, ['upper', 'lower', 'title', 'random']))
        self.case_type = case_type

    def __repr__(self):
        return 'Case' + '-' + self.case_type

    def _transform(self, sample, field='x', n=1, **kwargs):
        """ Transform each sample case according field.

        Args:
            sample: dict
                input data, normally one data sample.
            field: str
                indicate which filed to transform

        Returns:
            dict.
                transformed sample list.
        """

        field_value = sample.get_value(field)

        if isinstance(field_value, list):
            text = self.processor.inverse_tokenize(field_value)
        else:
            text = field_value
        assert isinstance(text, str)

        if self.case_type == 'random':
            case_type = ['upper', 'lower', 'title'][random.randint(0, 2)]
        else:
            case_type = self.case_type

        if case_type == 'upper':
            transform_text = text.upper()
        elif case_type == 'lower':
            transform_text = text.lower()
        else:
            transform_text = text.title()

        return [sample.replace_field(field, transform_text)]


if __name__ == "__main__":
    from TextRobustness.component.sample import SASample

    sent1 = 'The quick brown fox jumps over the lazy dog.'
    data_sample = SASample({'x': sent1, 'y': "negative"})

    trans_up = Case(case_type='upper')
    x = trans_up.transform(data_sample, n=1)
    print(x[0].dump())

    trans_low = Case(case_type='lower')
    x = trans_low.transform(data_sample, n=1)
    print(x[0].dump())

    trans_title = Case(case_type='title')
    x = trans_title.transform(data_sample, n=1)
    print(x[0].dump())

    trans_title = Case(case_type='random')
    for i in range(5):
        print(trans_title.transform(data_sample, n=1)[0].dump())

