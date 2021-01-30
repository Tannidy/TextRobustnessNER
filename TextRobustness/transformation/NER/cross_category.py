"""
Entity Swap by swaping entities with ones that can be labeled by different labels.
==========================================================
"""

import random

from TextRobustness.component.sample.ner_sample import NerSample
from TextRobustness.transformation import Transformation
from TextRobustness.common.settings import CROSS_ENTITIES
from TextRobustness.common.utils.load import read_cross_entities


class CrossCategory(Transformation):
    """ Entity Swap by swaping entities with ones that can be labeled by different labels. """

    def __init__(self, res_path=None, label_list=None, **kwargs):
        super().__init__()
        self.label_list = label_list

        if not res_path:
            res_path = CROSS_ENTITIES
        self.word_dic = read_cross_entities(res_path)

    # todo, n
    def _transform(self, sample, n=1, **kwargs):
        """

        Args:
            sample: NerSample
                Data for augmentation. It can be list of data (e.g. list
                    of string or numpy) or single element (e.g. string or numpy)
            **kwargs:

        Returns: Augmented data

        """
        new_sample_list = []
        entity_in_seq = sample.dump()['entities']
        for entity in reversed(entity_in_seq):
            substitude = random.choice(self.word_dic[entity['tag']])
            new_sample_list.append(sample.entity_replace(entity['start'], entity['end'], substitude, entity['tag']))

        return new_sample_list


if __name__ == "__main__":
    # data_path = "data/conll2003"
    sent1 = 'EU rejects German call to boycott British lamb .'
    data_sample = NerSample({'x': sent1, 'y': ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']})
    labels = [['ORG', 'PER', 'LOC', 'MISC'],
              ['ORG', 'LOC', 'PER', 'FAC', 'GPE', 'VEH', 'WEA'],
              ['ORG', 'LOC', 'PERSON', 'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW',
               'MONEY', 'NORP', 'ORDINAL', 'PERCENT', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']]
    label_list = labels[0]
    swap_ins = CrossCategory(label_list=label_list)
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
