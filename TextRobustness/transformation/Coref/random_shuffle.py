"""
Coref - Random shuffle: Randomly shuffle some sentences. 
At least (1-2*trans_p) sentences would be at the right pos, so don't worry 
==========================================================
"""

from math import ceil
from functools import reduce
from pprint import pprint
import random

from TextRobustness.transformation import Transformation
from TextRobustness.component.sample import CorefSample


class RandomShuffle(Transformation):
    """ Randomly change the position of some sentences

    Attributes:
        trans_p: proportion of deleted sentences; default 0.2
        processor: TextRobustness.common.preprocess.TextProcessor.

    """

    def __init__(self, trans_p=0.2, **kwargs):
        super().__init__()
        self.trans_p = trans_p

    @staticmethod
    def make_up_sample_with_shuffled_index(c1, sen_idxs):
        """ Given a CorefSample and shuffled sentence indexes, reproduce 
            a CorefSample with respect to the indexes. 
        Args:
            c1: a CorefSample. the original sample
                conll: a conll-style dict. same information of the original sample
            sen_idxs: a list of ints. the indexes to be preserved
        Returns:
            a CorefSample with respect to the shuffled index
        """
        conll = c1.dump()
        num_sentences = len(c1.sentences.field_value)
        lens_sentences = [len(sen) for sen in c1.sentences.field_value]

        def index_shift(word_idx):
            # belong to which sentence & original_shift
            ori_shift, sen_idx = 0, 0
            for j in range(num_sentences):
                ori_shift = ori_shift + lens_sentences[j]
                if ori_shift > word_idx:
                    ori_shift = ori_shift - lens_sentences[j]
                    sen_idx = j
                    break
            # shift after transformed
            shf_sen_idx = sen_idxs.index(sen_idx)
            shf_shift = 0
            for j in range(shf_sen_idx):
                shf_shift = shf_shift + lens_sentences[sen_idxs[j]]
            return ori_shift - shf_shift
        # make up the tfed sample
        # speakers, sentences
        speakers = [conll["speakers"][j] for j in sen_idxs]
        sentences = [conll["sentences"][j] for j in sen_idxs]
        # constituents, ner
        constituents = [
            [
                cn[0]-index_shift(cn[0]),
                cn[1]-index_shift(cn[0]),
                cn[2]]
            for cn in conll["constituents"]]
        ner = [
            [
                cn[0]-index_shift(cn[0]),
                cn[1]-index_shift(cn[0]),
                cn[2]]
            for cn in conll["ner"]]
        # clusters
        clusters = [
            [
                [
                    span[0] - index_shift(span[0]),
                    span[1] - index_shift(span[0])]
                for span in cluster]
            for cluster in conll["clusters"]]
        # composing output
        ret_conll = {
            "speakers": speakers,
            "doc_key": conll["doc_key"],
            "sentences": sentences,
            "constituents": constituents,
            "clusters": clusters,
            "ner": ner
        }
        return CorefSample(ret_conll)

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
            # shuffle: swap for trans_p * num_sentences (at least 1) times
            tfed_sen_idxs = list(range(num_sentences))
            for j in range(ceil(self.trans_p * num_sentences)):
                # randomly choose two sens ii & jj; then swap
                ii = int(random.random() * num_sentences)
                jj = int(random.random() * num_sentences)
                tmp = tfed_sen_idxs[ii]
                tfed_sen_idxs[ii] = tfed_sen_idxs[jj]
                tfed_sen_idxs[jj] = tmp
            # get the tfed sample and append to list
            sample_tfed = RandomShuffle.make_up_sample_with_shuffled_index(
                sample, tfed_sen_idxs)
            samples_tfed.append(sample_tfed)
        return samples_tfed


if __name__ == "__main__":
    from TextRobustness.component.sample.coref_sample import coref_sample1, coref_sample2, coref_sample3

    sample = reduce(
        CorefSample.concat_two_conlls, 
        [coref_sample1, coref_sample2, coref_sample3])
    # test
    model = RandomShuffle()
    samples_tfed = model.transform(sample)
    for s in samples_tfed:
        pprint(s.dump())
