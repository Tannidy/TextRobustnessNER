"""
SM Sample Class
============================================

"""

from TextRobustness.component.sample import Sample
from TextRobustness.component import Field
from TextRobustness.component.field import TextField

__all__ = ['SMSample']


class SMSample(Sample):
    def __init__(self, data, origin=None):
        super().__init__(data, origin=origin)

    def __repr__(self):
        return 'SMSample'

    def check_data(self, data):
        assert 'sentence1' in data and isinstance(data['sentence1'], str)
        assert 'sentence2' in data and isinstance(data['sentence2'], str)
        assert 'y' in data

    def load(self, data):
        """ Convert data dict which contains essential information to SMSample.

        Args:
            data: dict
                contains 'x', 'y' keys.
                'x' is a list [sentence1, sentence2]

        Returns:

        """
        self.sentence1 = TextField(data['sentence1'])
        self.sentence2 = TextField(data['sentence2'])
        self.y = Field(data['y'])

    def dump(self):
        return {'sentence1': self.sentence1.text,
                'sentence2': self.sentence2.text,
                'y': self.y.field_value}

