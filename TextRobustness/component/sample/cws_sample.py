"""
CWS Sample Class
============================================

"""

from TextRobustness.component.sample import Sample
from TextRobustness.component.field import TextField
from TextRobustness.component.field.cws_text_field import CWSTextField
from TextRobustness.common.preprocess.cws_text_processor import CWSTextProcessor

__all__ = ['CWSSample']


class CWSSample(Sample):
    """Our segmentation rules are based on ctb6.

        the input x can be a list or a sentence
        the input y is segmentation label include:B,M,E,S
        the y also can automatic generation,if you want automatic generation
            you must input an empty list and x must each word in x is separated by a space
            or split into each element of the list
        example:
            1. input {'x':'小明好想送Jo圣诞礼物', 'y' = ['B', 'E', 'B', 'E', 'S', 'B', 'E', 'B', 'E', 'B', 'E']}
            2. input {'x':['小明','好想送Jo圣诞礼物'], 'y' = ['B', 'E', 'B', 'E', 'S', 'B', 'E', 'B', 'E', 'B', 'E']}
            3. input {'x':'小明 好想 送 Jo 圣诞 礼物', 'y' = []}
            4. input {'x':['小明', '好想', '送', 'Jo', '圣诞', '礼物'], 'y' = []}
    """

    def __init__(self, data, origin=None):
        super().__init__(data, origin=origin)

    def __repr__(self):
        return 'CWSSample'

    def check_data(self, data):
        """Check the whether the data legitimate but we don't check that the label is correct
            if the data is not legal but acceptable format, change the format of data
        """
        assert 'x' in data and 'y' in data
        assert isinstance(data['y'], list)
        if not data['y']:
            data['x'] = data['x'].split(' ')
            sentence = ''
            for x in data['x']:
                x.replace(' ', '')
                sentence += x
                if len(x) == 1:
                    data['y'] += ['S']
                else:
                    data['y'] += ['B'] + ['M'] * (len(x) - 2) + ['E']
            data['x'] = sentence
        if isinstance(data['x'], list):
            sentence = ''
            for i in data['x']:
                i.replace(' ', '')
                sentence += i
            data['x'] = sentence
        assert isinstance(data['x'], str)
        cws_tag = ['B', 'M', 'E', 'S']
        for tag in data['y']:
            assert tag in cws_tag

    def load(self, data):
        """ Convert data dict which contains essential information to SASample.

        Args:
            data: dict
                contains 'x', 'y' keys.

        Returns:

        """
        self.x = CWSTextField(data['x'])
        self.y = TextField(data['y'])

    def update(self, x, y):
        """Update the x and y and new a sample.

        Args:
            x: new x
            y: new y

        Returns:
            new sample

        """
        sample = self.clone(self)
        sample.x = x
        sample.y = y
        return sample

    def dump(self):
        if isinstance(self.x.field_value, list):
            sentence = ''
            for word in self.x.field_value:
                sentence += word
            return {'x': sentence, 'y': self.y.field_value}
        return {'x': self.x.field_value, 'y': self.y.field_value}


if __name__ == '__main__':
    text = '张小明 好想 送 Jo 圣诞 礼物'
    label = []
    cws = CWSSample({'x': text, 'y': label})
    print(cws.y.field_value)
    print(cws.x.ner())
    print(cws.x.pos_tag())
