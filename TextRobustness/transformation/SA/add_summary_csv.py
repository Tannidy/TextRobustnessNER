"""
Add summaries according to the person or movies in the sentence from csv file
==========================================================
"""

from TextRobustness.component.sample import SASample
from TextRobustness.transformation import Transformation
from TextRobustness.common.settings import SA_PERSON_PATH
from TextRobustness.common.utils.load import sa_dict_loader


class AddSummaryCSV(Transformation):
    """Transforms an input by adding summaries of person and movies provided by csv.

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

    def _transform(self, sample, n=5, **kwargs):
        """ Transform text string .

        Args:
            sample: SASample
                input data, a Sample only contains 'x' field
            n: int
                number of generated samples
        Returns:
            list
                transformed sample list.
        """
        sub_list = sample.concat_token(self.max_name_len)
        insert_indices, insert_summaries = self._get_insert_info(sub_list)
        if not insert_indices:
            return []

        return [sample.insert_at_indices(insert_indices, insert_summaries)]

    def _get_insert_info(self, sub_list):
        """ Returns the index to insert the summary and the corresponding name.

        Args:
            sub_list: list
                A list including sub sentence of original sentence and corresponding indices
        Returns:
            indices: list
            summaries: list
        """
        summaries = []
        indices = []
        for item in sub_list:
            assert len(item) == 2
            name = item[0]
            index = item[1][1]

            if name in self.name_dict and index not in indices:
                indices.append(index-1)
                summaries.append(self.name_dict[name])
        return indices, summaries


if __name__ == "__main__":
    sent1 = "a classic of Paul Hartmann's kung fu cinema"
    data_sample = SASample({'x': sent1, 'y': "negative"})
    person_name_dict, person_max_name_len = sa_dict_loader(SA_PERSON_PATH)
    swap_ins = AddSummaryCSV(name_dict=person_name_dict, max_name_len=person_max_name_len)
    x = swap_ins.transform(data_sample, n=5)

    for sa_sample in x:
        print(sa_sample.dump())
