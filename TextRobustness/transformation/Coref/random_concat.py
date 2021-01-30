"""
Coref - Random concat: Concat randomly chosen samples from 
    `other_samples` behind samples from `sample`
============================================

"""

from pprint import pprint
import random


from TextRobustness.transformation import Transformation
from TextRobustness.component.sample import CorefSample


class RandomConcat(Transformation):
    """ Concatenate one extra sample to the original sample, with maintaining
        the coref-relations themselves.

    Attributes:
        processor: TextRobustness.common.preprocess.TextProcessor.

    """

    def __init__(self, **kwargs):
        super().__init__()

    def _transform(self, sample, n=5, **kwargs):
        """ 
        Args:
            sample: a CorefSample
            fields: Not used
            n: int; number of generated samples
            samples_other(optional): list of dict
                `samples_other` contains some other CorefSamples that also
                originate from conll-style dicts.
        Returns:
            samples_tfed: list
                transformed sample list.
        """
        samples_other = kwargs['samples_other']
        samples_tfed = []
        for i in range(n):
            # randomly choose a sample from samples_other
            j = int(random.random() * len(samples_other))
            # get the tfed sample and append to list
            sample_tfed = CorefSample.concat_two_conlls(
                sample, samples_other[j])
            samples_tfed.append(sample_tfed)
        return samples_tfed


if __name__ == "__main__":
    from TextRobustness.component.sample.coref_sample import coref_sample1, coref_sample2, coref_sample3

    sample = coref_sample1
    samples_other = [coref_sample2, coref_sample3]
    # test
    model = RandomConcat()
    samples_tfed = model.transform(sample, samples_other=samples_other)
    for s in samples_tfed:
        pprint(s.dump())
