from TextRobustness.component.sample import NLISample
from TextRobustness.transformation import Transformation
from TextRobustness.common.utils.num_word import _get_contradictory_hypothesis

LOWER_YEAR_NUM = 1000
UPPER_YEAR_NUM = 2020


class NliNumWord(Transformation):
    """Transforms an input by replacing its number word"""

    def __init__(self):
        super().__init__()

    def transform(self, sample, n=1, **kwargs):
        return self._transform(sample, **kwargs)

    def _transform(self, sample, n=1,  **kwargs):
        """

        Args:
            sample: dict:
            {
                "hypothesis": hypothesis,
                "premise": premise,
                "y": label
            }
            **kwargs:

        Returns: return a NLISample dict:
        {
            "hypothesis": hypothesis,
            "premise": premise,
            "y": "contradiction"
        }
        """

        sample = sample.dump()

        original_text = sample['premise']
        tokens = original_text.strip().split()
        flag = False
        for num, token in enumerate(tokens):
            try:
                number = int(token)
                if LOWER_YEAR_NUM <= number <= UPPER_YEAR_NUM:
                    continue
                #ent_hyp = _get_entailed_hypothesis(tokens, num, number)
                cont_hyp = _get_contradictory_hypothesis(tokens, num, number)
                flag = True
                break
            except:
                continue

        if not flag:
            return None

        transform_texts = NLISample({'hypothesis': original_text,
                                     'premise': cont_hyp,
                                     'y': 'contradiction'})

        return transform_texts


if __name__ == "__main__":

    sent1 = 'Mr Zhang has 10 students in Fudan university.'
    sent2 = 'Mr Zhang has 10 students.'
    sample1 = NLISample({
        'hypothesis': sent1,
        'premise': sent2,
        'y': '0'
    })
    ins = NliNumWord()
    trans = ins.transform(sample1)
    print(trans.dump())