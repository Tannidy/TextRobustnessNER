"""
POS Sample Class
============================================

"""

from TextRobustness.component.sample import Sample
from TextRobustness.component.field import ListField

__all__ = ['POSSample']


class POSSample(Sample):
    def __init__(self, data, origin=None):
        super().__init__(data, origin=origin)

    def __repr__(self):
        return 'POSSample'

    def check_data(self, data):
        assert 'x' in data
        assert 'y' in data

    def load(self, data):
        """ Convert data dict which contains essential information to SASample.

        Args:
            data: dict
                contains 'x', 'y' keys.

        Returns:

        """
        self.x = ListField(data['x'])
        self.y = ListField(data['y'])
        self.x_mask = ListField([1] * len(data['x']))

    def dump(self):
        return {'x': self.x.field_value,
                'y': self.y.field_value,
                'x_mask': self.x_mask.field_value}
