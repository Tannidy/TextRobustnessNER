"""
ABSA Sample Class
============================================

"""

from TextRobustness.component.sample import Sample
from TextRobustness.component import Field
from TextRobustness.component.field import TextField

__all__ = ['ABSASample']


class ABSASample(Sample):
    def __init__(self, data, origin=None):
        super().__init__(data, origin=origin)

    def __repr__(self):
        return 'ABSASample'

    def check_data(self, data):
        assert 'x' in data and isinstance(data['x'], str)
        assert 'y' in data and isinstance(data['y'], list)
        assert 'term_list' in data and isinstance(data['term_list'], dict)

    def load(self, data):
        """ Convert data dict which contains essential information to SASample.

        Args:
            data: dict
                contains 'x', 'y', 'aux_x' keys.

        Returns:

        """
        self.data = data
        self.x = Field(self.data['x'])
        self.y = Field(data['y'])
        self.dataset = Field(data['dataset'])
        self.term_list = Field(data['term_list'])
        if self.x.words[0].isupper():
            pass
        else:
            self.x.words[0] = self.x.words[0].lower()
        # TODO, add other info

    def dump(self):
        return {
                'x': self.data['x'],
                'y': self.y.field_value,
                'dataset': self.dataset.field_value,
                'term_list': self.term_list.field_value,
        }

