"""
Coref - Random delete: For one sample, randomly delete some sentences  
    of it
==========================================================
"""

from functools import reduce
from pprint import pprint
import random

from TextRobustness.transformation import Transformation
from TextRobustness.component.sample import CorefSample


class RandomDelete(Transformation):
    """ Randomly delete some sentences

    Attributes:
        trans_p: proportion of deleted sentences; default 0.2
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
            # randomly choose sentences to preserve
            preserved_sen_idxs = []
            for j in range(num_sentences):
                if random.random() > self.trans_p:
                    preserved_sen_idxs.append(j)
            # at least preserve 1 sen; at least delete 1 sen
            if len(preserved_sen_idxs) == 0: preserved_sen_idxs = [0]
            if len(preserved_sen_idxs) == num_sentences:
                j = int(random.random() * num_sentences)
                preserved_sen_idxs = preserved_sen_idxs[:j] + preserved_sen_idxs[j+1:]
            # get the tfed sample
            sample_tfed = CorefSample.part_conll(sample, preserved_sen_idxs)
            # post process: remove invalid clusters
            sample_tfed = CorefSample.remove_invalid_corefs_from_part(sample_tfed)
            # append to list
            samples_tfed.append(sample_tfed)
        return samples_tfed


if __name__ == "__main__":
    from TextRobustness.component.sample.coref_sample import coref_sample1, coref_sample2, coref_sample3

    sample = reduce(
        CorefSample.concat_two_conlls, 
        [coref_sample1, coref_sample2, coref_sample3])
    # test
    model = RandomDelete()
    samples_tfed = model.transform(sample)
    for s in samples_tfed:
        pprint(s.dump())
