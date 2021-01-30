"""
Add the difference part of target in ABSA task
==========================================================
"""

import random
import string
import language_tool_python
from nltk.tree import ParentedTree
from allennlp.predictors.predictor import Predictor
from TextRobustness.component.sample import ABSASample
from TextRobustness.common.utils.load import absa_dict_loader
from TextRobustness.common.settings import ABSA_DATA_PATH, ABSA_CONSTITUENT_PATH
from TextRobustness.transformation.ABSA.absa_transformation import ABSATransformation


class AbsaAddDiff(ABSATransformation):
    """Add the difference part of aspect to the end of original sentence.
    The difference part is extracted from the training set of SemEval2014.

    Attributes:
        language: default english
    """
    def __init__(self, language="eng"):
        super().__init__()

        if language is not "eng":
            raise ValueError(f"Language {language} is not available.")
        self.language = language
        self.language_tool = language_tool_python.LanguageTool('en-US')
        self.predictor = Predictor.from_path(ABSA_CONSTITUENT_PATH)

    def _transform(self, sample, **kwargs):
        """Transform data sample to a list of Sample.

        Args:
            sample->ABSASample instance:
            {
                'x': str
                    input sentence
                'y': List[str]
                    input label
                'dataset': str
                    'restaurant' or 'laptop' dataset in SemEval2014
                'term_list': dict
                    {
                        'term': list
                            aspect term
                        'id': int
                            sentence id
                        'from': int
                            the start of aspect term
                        'to': int
                            the end of aspect term
                        'polarity': str
                            polarity of aspect
                        'opinion_words': str
                            opinion words of aspect
                        'opinion_position': list
                            [start of opinion words, end of opinion words]
                    }
            }
        Returns:
            trans_samples: List[dict]
                List of transformed sample dict, its format
                is similar as input sample.
        """
        trans_samples = []
        examples = absa_dict_loader(ABSA_DATA_PATH, sample.dataset)
        positive_text, negative_text, neutral_text = self.get_extra_text(examples)
        term_list = sample.term_list
        all_term = [term_list[idx]['term'] for idx in term_list]

        for term_id in term_list:
            sentence = sample.x
            term_from = term_list[term_id]['from']
            term_to = term_list[term_id]['to']
            term = term_list[term_id]['term']
            polarity = term_list[term_id]['polarity']

            if polarity == 'positive':
                add_text = negative_text
            elif polarity == 'negative':
                add_text = positive_text
            else:
                add_text = neutral_text

            add_sentence = get_add_sentence(add_text, all_term, sentence)
            trans_sentence = self.concatenate_sentence(sentence, add_sentence)

            trans_sample = {
                'x': trans_sentence,
                'y': [polarity],
                'dataset': sample.dataset,
                'term_list': {
                    'term': term,
                    'id': term_id,
                    'from': term_from,
                    'to': term_to,
                    'polarity': polarity,
                }
            }
            trans_sample = ABSASample(trans_sample)
            trans_samples.append(trans_sample.dump())
        return trans_samples

    def get_extra_text(self, examples):
        """Get the extra text from SemEval2014 training dataset.

        Args:
            examples: dict
        Returns:
            positive_text: list
            negative_text: dict
            neutral_text: dict
        """
        positive_text = []
        negative_text = []
        neutral_text = []
        for sentence_id in examples:
            sentence = examples[sentence_id]['sentence']
            term_list = examples[sentence_id]['term_list']
            annotations = self.get_constituent(sentence)
            for term_id in term_list:
                term = term_list[term_id]['term']
                term_polarity = term_list[term_id]['polarity']
                term_opinion = term_list[term_id]['opinion_words'][-1].lower()
                try:
                    # some sentence can not be parsed
                    ptree = ParentedTree.fromstring(annotations)
                except Exception:
                    continue
                phrases = self.get_phrase(term, term_opinion, ptree)
                extra_sentence = get_extra_sentence(term_list, term_id, phrases)
                if len(extra_sentence) == 0:
                    continue
                extra_sentence = extra_sentence[0]
                if term_polarity == 'positive':
                    positive_text.append((term.lower(), [s.lower() for s in extra_sentence]))
                elif term_polarity == 'negative':
                    negative_text.append((term.lower(), [s.lower() for s in extra_sentence]))
                elif term_polarity == 'neutral':
                    neutral_text.append((term.lower(), [s.lower() for s in extra_sentence]))
        return positive_text, negative_text, neutral_text

    def concatenate_sentence(self, sentence, add_sentence):
        """Concatenate the extra part to original sentence.

        Args:
            sentence: list
            add_sentence: list
        Returns:
            trans_sentence: list
        """
        opi_tag = self.get_postag(add_sentence, 0, 1)
        if opi_tag[0] != 'CONJ':
            tmp_sentence = 'but ' + self.untokenize(add_sentence)
            matches = self.language_tool.check(tmp_sentence)
            trans_sentence = language_tool_python.utils.correct(tmp_sentence, matches)
            trans_sentence = trans_sentence[4:]

            if 'but' in sentence or 'although' in sentence:
                trans_sentence = sentence + "; " + trans_sentence
            else:
                trans_sentence = sentence + ", but " + trans_sentence
        else:
            tmp_sentence = self.untokenize(add_sentence)
            matches = self.language_tool.check(tmp_sentence)
            trans_sentence = language_tool_python.utils.correct(tmp_sentence, matches)
            trans_sentence = sentence + ". " + trans_sentence[
                0].upper() + trans_sentence[1:]
        return trans_sentence

    def get_constituent(self, x):
        """Get constituent parser.

        Args:
            x: list
        Returns:
            annotations: list
        """
        annotations = self.predictor.predict(sentence=x)['trees']
        return annotations


def get_extra_sentence(term_list, term_id, phrases):
    """Get the extra sentence from phrases text.

    Args:
        term_list: dict
        term_id: str
        phrases: list
    Returns:
        extra_sentence: list
    """
    other_terms = []
    extra_sentence = []
    for other_id in term_list:
        if other_id != term_id:
            other_terms.append(
                ''.join(term_list[other_id]['term'].split(' ')))
    for phrase in phrases:
        phrase_ = ''.join(phrase)
        overlap = False
        for other_term in other_terms:
            if other_term in phrase_:
                overlap = True
                break
        if not overlap:
            extra_sentence.append(phrase)
    extra_sentence = sorted(extra_sentence, key=len)
    return extra_sentence


def get_add_sentence(add_text, all_term, sentence):
    """Get the sentence that owns different polarity compared with
    the aspect. Choose 1~3 sentences randomly from add_text and put
    them together.

    Args:
        add_text: dict
        all_term: str
        sentence: list
    Returns:
        add_sentence: list
    """
    punctuation = '.'
    if sentence[-1] == string.punctuation:
        punctuation = sentence[-1]
    while True:
        add_sentence_num = random.randint(1, 3)
        random_num1, random_num2, random_num3 = random.sample(range(len(add_text)), 3)
        random_sentence1 = add_text[random_num1]
        random_sentence2 = add_text[random_num2]
        random_sentence3 = add_text[random_num3]
        if random_sentence1[0] not in all_term and random_sentence2[
            0] not in all_term and random_sentence3[
            0] not in all_term:
            break

    if add_sentence_num == 3:
        add_sentence = random_sentence1[1] + [','] + random_sentence2[1] + [
            'and'] + random_sentence3[1] + [punctuation]
    elif add_sentence_num == 2:
        add_sentence = random_sentence1[1] + ['and'] + random_sentence2[1] + [
            punctuation]
    else:
        add_sentence = random_sentence1[1] + [punctuation]
    return add_sentence


if __name__ == "__main__":
    absa_sample = {
        "x": "Great food, great waitstaff, great atmosphere, and best of all GREAT beer!",
        "y": ["positive", "positive", "positive", "positive"],
        "dataset": "restaurant",
        "term_list": {
            "11302355#533813#0_3": {
                "id": "11302355#533813#0_3",
                "polarity": "positive",
                "term": "food",
                "from": 6,
                "to": 10,
                "opinion_words": ["Great"],
                "opinion_position": [[0, 5]]
            },
            "11302355#533813#0_1": {
                "id": "11302355#533813#0_1",
                "polarity": "positive",
                "term": "waitstaff",
                "from": 18,
                "to": 27,
                "opinion_words": ["great"],
                "opinion_position": [[12, 17]]
            },
            "11302355#533813#0_0": {
                "id": "11302355#533813#0_0",
                "polarity": "positive",
                "term": "atmosphere",
                "from": 35,
                "to": 45,
                "opinion_words": ["great"],
                "opinion_position": [[29, 34]]
            },
            "11302355#533813#0_2": {
                "id": "11302355#533813#0_2",
                "polarity": "positive",
                "term": "beer",
                "from": 69,
                "to": 73,
                "opinion_words": ["best", "GREAT"],
                "opinion_position": [[51, 55], [63, 68]]
            }
        },
    }

    data_sample = ABSASample(absa_sample)
    reverse_target = AbsaAddDiff()
    trans = reverse_target.transform(data_sample)
    print(trans)
