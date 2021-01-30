"""
Entity Swap by OOV entities.
==========================================================
"""

import random

from TextRobustness.common.settings import NER_OOV_ENTITIES
from TextRobustness.common.utils.load import load_oov_entities
from TextRobustness.component.sample.ner_sample import NerSample
from TextRobustness.transformation import Transformation


class OOV(Transformation):
    """Entity Swap by swaping entities with ones that haven't appeared in the training dataset.

    """
    def __init__(self, res_path=None, **kwargs):
        super().__init__()
        self.res_path = res_path
        self.oov_dic = load_oov_entities(NER_OOV_ENTITIES)

    # TODO, n 个样本如何返回？
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
            substitude = random.choice(self.oov_dic[entity['tag']])
            new_sample_list.append(sample.entity_replace(entity['start'], entity['end'], substitude, entity['tag']))

        return new_sample_list


if __name__ == "__main__":
    # data_path = "data/conll2003"
    sent1 = 'EU rejects German call to boycott British lamb .'
    data_sample = NerSample({'x': sent1, 'y': ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']})
    swap_ins = OOV()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
