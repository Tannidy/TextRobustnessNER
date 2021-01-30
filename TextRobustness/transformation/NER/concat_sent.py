"""
Concatenate sentences to a longer one.
==========================================================
"""
import json
import os
import random

from TextRobustness.common.settings import DATA_PATH
from TextRobustness.component.sample.ner_sample import NerSample
from TextRobustness.transformation import Transformation


# TODO, 现有框架怎么实现？
class ConcatSent(Transformation):
    """Concatenate sentences to a longer one.
     Attributes:
         res_path: string
            dir for vocab/dict
     """
    def __init__(self, length_threshold, **kwargs):
        super().__init__()
        self.length_threshold = length_threshold

    def _transform(self, sample, n=1, **kwargs):
        '''Transform data sample to a list of Sample.
        Args:
            input_sample: NerSample
                Data sample for augmentation.
            n: int
                Default is 5. MAx number of unique augmented output.
            **kwargs: another sample

        Returns: Augmented data

        '''
        another_sample = kwargs['another_sample']
        new_sample_list = []
        new_sample_list.append(sample.concat_samples(another_sample))
        return new_sample_list

    def transform(self, sample, n=1, field='x', **kwargs):
        """Transform data sample to a list of Sample.

        Args:
            sample: Sample
                Data sample for augmentation.
            n: int
                Max number of unique augmented output, default is 5.
            field: str
                Indicate which fields to apply transformations.
            kwargs: dict
                other auxiliary params.
        Returns:
             Augmented data: list of Sample

        """

        transform_results = self._transform(sample, n=n, field=field, **kwargs)

        return transform_results

if __name__ == "__main__":
    sent1 = 'EU rejects German call to boycott British lamb .'
    data_sample1 = NerSample({'x': sent1, 'y': ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']})
    data2 = {'x': ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'],
            'y': ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']}
    data_sample2 = NerSample(data2)
    swap_ins = ConcatSent(50)
    x = swap_ins.transform(data_sample1, n=1, another_sample=data_sample2)

    for sample in x:
        print(sample.dump())
