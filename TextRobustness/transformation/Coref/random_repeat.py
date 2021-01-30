"""
Coref - Random repeat: Randomly choose some sentences, and each of them
    will be repeated somewhere else in the sample.
==========================================================
"""

from functools import reduce
from math import ceil
from pprint import pprint
import random

from TextRobustness.transformation import Transformation
from TextRobustness.component.sample import CorefSample


class RandomRepeat(Transformation):
    """ Randomly choose some sentences, and each of them will be repeated 
        somewhere else in the sample.

    Attributes:
        trans_p: proportion of repeated sentences; default 0.2
        processor: TextRobustness.common.preprocess.TextProcessor.

    """

    def __init__(self, trans_p=0.2, **kwargs):
        super().__init__()
        self.trans_p = trans_p

    def _transform(self, sample, n=5, **kwargs):
        """ 
        Args:
            sample: a CorefSample
            fields: Not used
            n: int; number of generated samples
        Returns:
            samples_tfed: list
                transformed sample list.
        """
        num_sentences = len(sample.sentences)
        samples_tfed = []
        for i in range(n):
            sample_tfed = CorefSample(sample.dump())
            # repeat times: trans_p * num_sentences; at least 1
            for j in range(ceil(num_sentences * self.trans_p)):
                # randomly choose the sentence to repeat
                ori_sen_idx = int(random.random() * (num_sentences))
                s_pt = CorefSample.part_conll(sample, [ori_sen_idx])
                # randomly choose tfed_sen_idx: 
                # k_sen will be inserted after position tfed_sen_idx
                # tfed_sen_idx in [0, num_sentences + j - 1): 
                # tfed_sen_idx cannot be the last one
                assert len(sample_tfed.sentences) == num_sentences + j
                tfed_sen_idx = int(random.random() * (num_sentences + j - 1))
                # tfed[:tfed_sen_idx]+k_pt+tfed[tfed_sen_idx:]
                sample_tfed_pt1 = CorefSample.part_before_conll(
                    sample_tfed, tfed_sen_idx+1)
                sample_tfed_pt2 = CorefSample.part_after_conll(
                    sample_tfed, tfed_sen_idx+1)
                sample_tfed = CorefSample.concat_conll_parts(
                    sample_tfed_pt1, s_pt, sample_tfed_pt2)
            # get the tfed sample and append to list
            samples_tfed.append(sample_tfed)
        return samples_tfed


if __name__ == "__main__":
    from TextRobustness.component.sample.coref_sample import coref_sample1, coref_sample2, coref_sample3

    sample = reduce(
        CorefSample.concat_two_conlls, 
        [coref_sample1, coref_sample2, coref_sample3])
    # test
    model = RandomRepeat()
    samples_tfed = model.transform(sample)
    for s in samples_tfed:
        pprint(s.dump())
