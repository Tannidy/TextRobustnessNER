"""
NLI Sample Class
============================================

"""

from TextRobustness.component.sample import Sample
from TextRobustness.component import Field
from TextRobustness.component.field import TextField

__all__ = ['NLISample']


class NLISample(Sample):
    def __init__(self, data, origin=None):
        super().__init__(data, origin=origin)

    def __repr__(self):
        return 'NLISample'

    def check_data(self, data):
        assert 'hypothesis' in data and isinstance(data['hypothesis'], str)
        assert 'premise' in data and isinstance(data['premise'], str)
        assert 'y' in data

    def load(self, data):
        """ Convert data dict which contains essential information to NLISample.

        Args:
            data: dict
                contains 'x', 'y' keys.
                'x' is a list [hypothesis, premise]

        Returns:

        """
        self.hypothesis = TextField(data['hypothesis'])
        self.premise = TextField(data['premise'])
        self.y = Field(data['y'])

    def dump(self):
        return {'hypothesis': self.hypothesis.text,
                'premise': self.premise.text,
                'y': self.y.field_value}

