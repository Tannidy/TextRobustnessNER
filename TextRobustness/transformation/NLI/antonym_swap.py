from TextRobustness.component.sample import NLISample
from TextRobustness.transformation import Transformation
from TextRobustness.common.utils.word_net import lesk
from TextRobustness.common.settings import BLACK_LIST_WORD


class NliAntonymSwap(Transformation):
    """ Transforms an input by replacing its words with antonyms provided by
        WordNet. Download nltk_data before running.
        Implement follow by Stress Test Evaluation for Natural Language Inference
        For the correctness of trasformation we swap the word has best_sense(Wordnet) to its antonym

    """
    def __init__(self, language="eng"):
        super().__init__()
        self.language = language
        self.blacklist_words = BLACK_LIST_WORD

    def transform(self, sample, n=1, **kwargs):
        return self._transform(sample, **kwargs)

    def _transform(self, sample, **kwargs):
        """
        Args:
            sample: dict:
            {
                "hypothesis": hypothesis,
                "premise": premise,
                "y": label
            }
            **kwargs:

        Returns: the NLISample
            {
                "hypothesis": hypothesis,
                "premise": premise,
                "y": 'contradiction'
            }
        which hypothesis is the original hypothesis
        and hypothesis is the premise that swap the key word into the antonym.
        For the correctness this method only transform the sample whose label_tag is entailment.

        """
        sample = sample.dump()
        label_tag = sample['y']

        if label_tag != 'entailment':
            return None

        original_text1 = sample["hypothesis"]
        original_text2 = sample["premise"]
        tokens1 = original_text1.strip().split()
        tokens2 = original_text2.strip().split()
        trans_sample = None
        for num, each_word in enumerate(tokens2):
            if each_word not in self.blacklist_words:
                best_sense = lesk(tokens2, each_word)
                if best_sense is not None and (best_sense.pos() == 's' or best_sense.pos() == 'n'):
                    for lemma in best_sense.lemmas():
                        possible_antonyms = lemma.antonyms()
                        for antonym in possible_antonyms:
                            if "_" in antonym._name or antonym._name == "civilian":
                                continue
                            if each_word not in tokens1:
                                continue
                            new_s1 = original_text2.replace(each_word, antonym._name, 1)
                            trans_text = {
                                'hypothesis': new_s1,
                                'premise': original_text2,
                                'y': 'contradiction'
                            }
                            trans_sample = NLISample(trans_text)
        return trans_sample


if __name__ == "__main__":
    sent1 = "The quick brown fox jumps over the lazy dog ."
    sent2 = "The quick brown fox jumps over the lazy dog ."
    data_sample = NLISample({
        'hypothesis': sent1,
        'premise': sent2,
        'y': '1'
    })
    ins = NliAntonymSwap()
    sample1 = ins.transform(data_sample)
    print(sample1.dump())
