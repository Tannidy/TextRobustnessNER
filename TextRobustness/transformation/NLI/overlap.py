from TextRobustness.component.sample import NLISample
from TextRobustness.transformation import Transformation
from TextRobustness.common.utils.overlap_templates import *


def no_the(sentence):
    return sentence.replace("the ", "")


def repeaters(sentence):
    condensed = no_the(sentence)
    words = []

    for word in condensed.split():
        if word in lemma:
            words.append(lemma[word])
        else:
            words.append(word)

    if len(list(set(words))) == len(words):
        return False
    else:
        return True


class NliOverlap(Transformation):
    """Generate some samples by templates
       implement follow
       Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference ACL2019
       In order to generate some sample whose premise is the sequence of the hypothesis but the semantic are different.
       exmaple:
       {
            hypothesis: I hope Tom can go to school.
            premise: Tom go to school.
            label: non-entailment
       }
    """
    def __init__(self):
        super().__init__()

    def transform(self, sample1, n=5, **kwargs):
        return self._transform(n, **kwargs)

    def _transform(self, n=5, **kwargs):
        """
        Args:
            n: this method will generate n samples for every templates directly.
            **kwargs:

        Returns: A list of SMSample dict :
        {
            "x": [sentence1, sentence2],
            "y": label (entailment or non-entailment)
        }
        """

        example_counter = 0
        trans_list = []
        for template_tuple in template_list:
            label = template_tuple[2]
            template = template_tuple[3]

            example_dict = {}
            count_examples = 0

            while count_examples < n:
                example = template_filler(template)

                example_sents = tuple(example[:2])

                if example_sents not in example_dict and not repeaters(example[0]):
                    example_dict[example_sents] = 1
                    trans_sample = {
                        'hypothesis': example[0],
                        'premise': example[1],
                        'y': label
                    }
                    trans_list.append(NLISample(trans_sample))
                    count_examples += 1
                    example_counter += 1

        return trans_list


if __name__ == "__main__":

    overlap_ins = NliOverlap()
    trans_list = overlap_ins.transform(n=1)
    for sample in trans_list:
        print(sample.dump())