"""
Word Swap by swapping names according to the person or movies in the sentence from csv file
==========================================================
"""
from TextRobustness.component.sample import SASample
from TextRobustness.transformation import Transformation
from TextRobustness.common.settings import SA_PERSON_PATH
from TextRobustness.common.utils.load import sa_dict_loader
import random


class WordSwapCSV(Transformation):
    """Transforms an input by replacing its words with other names provided by csv.

    Attributes:
        name_dict: dict
            A dict to complete the replacement, the key of dict is name and value is the corresponding summary
        max_name_len: int
            The longest length of a name in the name_dict
    """

    def __init__(self, name_dict, max_name_len, **kwargs):
        super().__init__()
        self.name_dict = name_dict
        self.max_name_len = max_name_len

    def _transform(self, sample, field='x', n=5, **kwargs):
        """ Transform text string according field.

        Args:
            sample: dict
                input data, normally one data component.
            fields: str or list
                indicate which fields to transform,
                for multi fields , substitute them at the same time.
            n: int
                number of generated samples
        Returns:
            list
                transformed sample list.
        """

        # To speed up the query, dividing the original sentence into n-tuple string
        sub_list = sample.concat_token(self.max_name_len)
        swap_indices, swap_tokens = self._get_swap_info(sub_list)
        if not swap_indices:
            return []

        trans_samples = [sample.replace_fields_at_indices(['x'], [swap_indices], [swap_tokens])]
        return trans_samples

    def _get_swap_info(self, sub_list):
        """Get the indices of the words to be replaced and the new names to replace them

        Args:
            sub_list: list

        Returns:
            indices: list
                indices of tokens that should be replaced
            names: list
                The names that correspond to indices and is used to replace them
        """

        def check_collision(index, r):
            for i, range1 in enumerate(r):
                l1, r1 = range1
                l2, r2 = index
                if max(l1, l2) < min(r1, r2):
                    return True
            return False

        indices = []
        names = []
        for item in sub_list:
            current_name = item[0]
            current_index = item[1]
            if current_name in self.name_dict and not check_collision(current_index, indices):
                indices.append(current_index)
                names.append(random.sample(self.name_dict.keys(), 1))
        return indices, names


if __name__ == "__main__":
    sent1 = "a classic of Paul Hartmann's kung fu cinema,"
    data_sample = SASample({'x': sent1, 'y': "negative"})
    person_name_dict, movie_max_name_len = sa_dict_loader(SA_PERSON_PATH)
    swap_ins = WordSwapCSV(name_dict=person_name_dict, max_name_len=movie_max_name_len)
    x = swap_ins.transform(data_sample, n=5)

    for x_sample in x:
        print(x_sample.dump())
