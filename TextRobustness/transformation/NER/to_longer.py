"""
Substitute short entities to longer ones
==========================================================
"""
import json
import random

from TextRobustness.common.settings import LONG_ENTITIES
from TextRobustness.component.sample.ner_sample import NerSample
from TextRobustness.transformation import Transformation


class ToLonger(Transformation):
    """Swap entities which shorter than threshold to longer ones.

    Attributes:
        max_rep_length: maximum length of entity to apply substitution.
        long_entities_dic: dict to hold long entity of various types.
    """
    def __init__(self, max_rep_length=2, res_path=None, **kwargs):
        super().__init__()
        if not res_path:
            res_path = LONG_ENTITIES
        self.max_rep_length = max_rep_length
        self.long_entities_dic = json.load(open(res_path, 'r'))

    # TODO， n个样本返回怎么实现
    def _transform(self, sample, n=1, **kwargs):
        """Transform data sample to a list of Sample.
        Args:
            input_sample: NerSample
                Data sample for augmentation.
            n: int
                Default is 1. MAx number of unique augmented output.
            **kwargs:

        Returns:
            Augmented data

        """
        new_sample_list = []
        entity_in_seq = sample.get_value('entities')
        replaced_entities_list = []
        replaces_samples = [sample.clone()] * n

        for entity in reversed(entity_in_seq):
            replaced_entities = []
            cur_entity = entity['entity']

            if n > len(self.long_entities_dic[entity['tag']]):
                raise ValueError('Generate number ({0}) is large than entities dic ({1})!'
                                 .format(n, len(self.long_entities_dic[entity['tag']])))

            if len(cur_entity.split(" ")) <= self.max_rep_length:
                long_entities = random.sample(self.long_entities_dic[entity['tag']], n)
                replaced_entities.append(long_entities)

        if not replaced_entities_list:
            return []

        for i in range(n):
            for rep_entity in reversed(replaced_entities_list):
                replaces_samples[i] = replaces_samples[i].entity_replace(entity['start'], entity['end'],
                                                                         rep_entity, entity['tag'])

        return new_sample_list


if __name__ == "__main__":
    sent1 = 'EU rejects German call to boycott British lamb .'
    data_sample = NerSample({'x': sent1, 'y': ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']})
    swap_ins = ToLonger()
    x = swap_ins.transform(data_sample, n=5)

    for ner_sample in x:
        print(ner_sample.dump())
