"""
SA Sample Class
============================================

"""

from TextRobustness.component.sample import Sample
from TextRobustness.component import Field
from TextRobustness.component.field import TextField


__all__ = ['SASample']


class SASample(Sample):
    def __init__(self, data, origin=None):
        super().__init__(data, origin=origin)

    def __repr__(self):
        return 'SASample'

    def check_data(self, data):
        assert 'x' in data and isinstance(data['x'], str)
        assert 'y' in data

    def load(self, data):
        """ Convert data dict which contains essential information to SASample.

        Args:
            data: dict
                contains 'x', 'y' keys.

        Returns:

        """
        self.x = TextField(data['x'])
        self.y = Field(data['y'])

    def dump(self):
        return {'x': self.x.text, 'y': self.y.field_value}

    def insert_at_indices(self, indices, new_items):
        """Insert new_items after the indices of x.

        Args:
            indices: list
                indices of tokens that should be replaced
            new_items: list


        Returns:
            sample: SASample
                new sample with transformed filed 'x'
        """
        sample = self.clone(self)
        items = []

        for index, item in zip(indices, new_items):
            # TODO
            new_item = "%s {%s}" % (self.x.words[index], item)
            items.append(new_item)

        sample.x = sample.x.replace_at_ranges(indices, items)
        return sample

    def concat_token(self, max_name_len):
        """ Find all the n-tuple that may be a name and splice it into a string

        Args:
            max_name_len: int

        Returns:
            sub_list: list
                The list is composed of lists which the first dimension is n-tuples spliced into strings, and the second
                dimension is the index of n-tuples in the original sentence
        """
        tokens = self.get_value('x')
        sub_list = []
        for i in range(len(tokens)):
            current_str = tokens[i]
            sub_list.append([current_str, [i, i + 1]])
            for j in range(1, max_name_len):
                if i + j < len(tokens):
                    current_str += ' %s' % tokens[i + j]
                    sub_list.append([current_str, [i, i + j + 1]])
                else:
                    break
        return sub_list


if __name__ == "__main__":
    x = "this is a sentence aa bnn cc d"
    y = "positive"
    sa_sample = SASample({'x': x, 'y': y})
    new_sample = sa_sample.replace_field('x', "I cant believe this sentence!")
    print(new_sample.is_origin)
    print(new_sample.dump())

    new_sample = sa_sample.insert_at_indices([0, 2, 3, 5], ["0", "2", "3", "5"])
    print(new_sample.is_origin)
    print(new_sample.dump())
