from TextRobustness.component.sample import SMSample
from TextRobustness.transformation import Transformation
from TextRobustness.common.utils.num_word import _get_contradictory_hypothesis

LOWER_YEAR_NUM = 1000
UPPER_YEAR_NUM = 2020


class SmNumWord(Transformation):
    """Transforms an input by replacing its number word"""

    def __init__(self):
        super().__init__()

    def transform(self, sample, n=1, **kwargs):
        return self._transform(sample, **kwargs)

    def _transform(self, sample, **kwargs):
        """

        Args:
             sample: dict:
            {
                "sentence1": sentence1,
                "sentence2": sentence2,
                "y": label
            }
            **kwargs:

        Returns: the SMSample {"sentence1":sentence1,
                               "sentence2": sentence2],
                               "y":'0'
                               },

        """
        sample = sample.dump()

        original_text = sample['sentence2']
        tokens = original_text.strip().split()
        flag = False
        for num, token in enumerate(tokens):
            try:
                number = int(token)
                if LOWER_YEAR_NUM <= number <= UPPER_YEAR_NUM:
                    continue
                # ent_hyp = _get_entailed_hypothesis(tokens, num, number)
                cont_hyp = _get_contradictory_hypothesis(tokens, num, number)
                flag = True
                break
            except:
                continue

        if not flag:
            return None

        transform_texts = SMSample({'sentence1': original_text,
                                    'sentence2': cont_hyp,
                                    'y': '0'})

        return transform_texts


if __name__ == "__main__":
    sent1 = 'Mr Zhang has 10 students in Fudan university.'
    sent2 = 'Mr Zhang has 10 students.'
    sample1 = SMSample({
        'sentence1': sent1,
        'sentence2': sent2,
        'y': '0'
    })
    ins = SmNumWord()
    trans = ins.transform(sample1)
    print(trans.dump())