"""
Coref - Random Insert: some irrelevance sentences will be inserted
    into original sentences. These sentences include no corefs.
==========================================================
"""

from math import ceil
from pprint import pprint
import random

from TextRobustness.component.sample import CorefSample
from TextRobustness.transformation import Transformation


class RandomInsert(Transformation):
    """ RandomInsert: trans_p * num_sentences of irrelevance sentences are 
        sampled from samples_other, and they will be inserted into original 
        sentences in sample. These inserted sentences include no corefs.

    Attributes:
        trans_p: proportion of inserted sentences; default 0.2
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
            samples_other(optional): list of dict
                `samples_other` contains some other CorefSamples that also
                originate from conll-style dicts.
        Returns:
            samples_tfed: list
                transformed sample list.
        """
        samples_other = kwargs['samples_other']
        num_sentences = len(sample.sentences)
        num_clusters = len(sample.clusters)
        samples_tfed = []

        for i in range(n):
            sample_tfed = CorefSample(sample.dump())
            # insert times: trans_p * num_sentences; at least 1
            for j in range(ceil(num_sentences * self.trans_p)):
                # randomly choose the irrelevant sentence
                k = int(random.random() * len(samples_other))
                k_sen_idx = int(random.random() * len(samples_other[k].sentences))
                k_sen = samples_other[k].sentences[k_sen_idx]
                # make the part to concat
                k_pt_conll = {
                    "speakers": [["sp"] * len(k_sen)],
                    "doc_key": samples_other[k].doc_key.field_value,
                    "sentences": [k_sen],
                    "constituents": [],
                    "clusters": [[]] * num_clusters,
                    "ner": []
                }
                k_pt = CorefSample(k_pt_conll)
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
                    sample_tfed_pt1, k_pt, sample_tfed_pt2)
            # get the tfed sample and append to list
            samples_tfed.append(sample_tfed)

        return samples_tfed


if __name__ == "__main__":
    from TextRobustness.component.sample.coref_sample import coref_sample1, coref_sample2, coref_sample3

    sample = coref_sample1
    samples_other = [coref_sample2, coref_sample3]
    # test
    model = RandomInsert()
    samples_tfed = model.transform(sample, samples_other=samples_other)
    for s in samples_tfed:
        pprint(s.dump())
