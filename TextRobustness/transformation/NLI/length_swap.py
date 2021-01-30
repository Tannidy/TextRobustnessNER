from TextRobustness.component.sample import NLISample
from TextRobustness.transformation import Transformation


class NliLength(Transformation):
    """Adding the Meaningless sentences to the hypothesis and remain both premise and label.
        Users can use their own meaningless sentences by change the text in transform method"""
    def __init__(self):
        super().__init__()

    def transform(self, sample, n=1,
                  text=' And we have to say that the sun is not moon, the moon is not sun.', **kwargs):
        return self._transform(sample, text, n=1, **kwargs)

    def _transform(self, sample, text, n=1, **kwargs):
        """

        Args:
            sample: dict:
            {
                "hypothesis": hypothesis,
                "premise": premise,
                "y": label
            }
            text: the meaningless sentences
            **kwargs:

        Returns: A NLISample dict:
        {
            "hypothesis": hypothesis,
            "premise": premise,
            "y": label (no change)
        }

        """

        label_tag = sample['y']
        original_text1 = sample['hypothesis']
        original_text2 = sample['premise']
        original_text1 = original_text1 + text
        transform_texts = {
            'hypothesis': original_text1,
            'premise': original_text2,
            'y': label_tag
        }

        return NLISample(transform_texts)


if __name__ == "__main__":

    sent1 = 'Mr Zhang has 10 students in Fudan university.'
    sent2 = 'Mr Zhang has 10 students.'
    data_sample = {'hypothesis': sent1,
                   'premise': sent2,
                   'y': "entailment"}

    ins = NliLength()
    trans = ins.transform(data_sample)
    print(trans.dump())