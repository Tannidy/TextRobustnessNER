"""
Remove a subtree in the sentence
============================================
"""

from TextRobustness.transformation import Transformation


class RemoveSubtree(Transformation):
    """
    Transforms the input sentence by removing a subordinate clause.
    """
    def __repr__(self):
        return 'RemoveSubtree'

    def _transform(self, sample, n=5, **kwargs):
        """Transform each sample case.

        Args:
            sample: DPSample

        Returns:
            list of DPSample
                transformed sample list.

        """
        subtrees = self.find_subtree(sample)
        if not subtrees:
            return []
        result = []

        for i, subtree in enumerate(subtrees):
            if i >= n:
                break
            else:
                sample_mod = sample.clone(sample)
                index = subtree[0] - 1
                for i in range(self.get_difference(subtree)):
                    sample_mod = sample_mod.delete_field_at_index('x', index)
                result.append(sample_mod)

        return result

    def find_subtree(self, sample):
        """ Find all the subtrees that can be removed.

        Args:
            sample: DPSample

        Returns:
            list of tuples
                A list of the subtrees, long to short.

        """
        words = sample.get_words('x')
        heads = sample.get_value('head')
        punc = []
        subtrees = []

        for i, word in enumerate(words):
            if word in (',', '.'):
                punc.append(i + 1)
        if len(punc) == 1:
            return None

        for i in range(len(punc) - 1):
            start = punc[i]
            end = punc[i + 1]
            flag = True
            bracket = 0
            interval = list(range(start + 1, end))

            for i, head in enumerate(heads):
                if int(head) in interval and (i + 1) not in interval:
                    flag = False
                    break

            for word in words[start:end - 1]:
                if word in ('-LRB-', '-RRB-'):
                    bracket += 1
            if flag is True and bracket % 2 == 0:
                subtrees.append((start, end))

        return sorted(subtrees, key=self.get_difference, reverse=True)

    def get_difference(self, num_pair):
        return num_pair[1] - num_pair[0]


if __name__ == '__main__':
    from TextRobustness.common.res.DP_DATA.data_sample import sample
    rms = RemoveSubtree()
    samples = rms.transform(sample)
    for sample in samples:
        print(sample.dump())
