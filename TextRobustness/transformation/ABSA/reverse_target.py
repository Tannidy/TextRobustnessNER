"""
Reverse the polarity of target in ABSA task
==========================================================
"""

import random
from TextRobustness.component.sample import ABSASample
from TextRobustness.transformation.ABSA.absa_transformation import ABSATransformation


class AbsaReverseTarget(ABSATransformation):
    """ Transforms the polarity of target by replacing its opinion words
        with antonyms provided by WordNet or adding the negation that
        pre-defined in our negative word list.

    Attributes:
        language: default english
    """

    def __init__(self, language="eng"):
        super().__init__()
        if language is not "eng":
            raise ValueError(f"Language {language} is not available.")
        self.language = language
        self.tokenize = self.processor.word_tokenize

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
        sentence = sample.x
        words_list = sample.x.words
        term_list = self.tokenize_term_list(sample)

        for term_id in term_list:
            term = term_list[term_id]['term']
            trans_words, trans_polarity = self.trans_aspect_polarity(term_list[term_id], words_list)
            trans_words = self.trans_conjunction(term_list, term_id, trans_words, trans_polarity)
            trans_sentence = self.get_sentence(trans_words, sentence)
            aspect_from, aspect_to = self.get_term_span(trans_sentence, term)

            trans_sample = {
                'x': trans_sentence,
                'y': [trans_polarity],
                'term_list': {
                    'term': term,
                    'id': term_id,
                    'from': aspect_from,
                    'to': aspect_to,
                    'polarity': trans_polarity,
                    }
                }
            trans_sample = ABSASample(trans_sample)
            trans_samples.append(trans_sample.dump())
        return trans_samples

    @staticmethod
    def get_other_opinions(term_to_position_list, term_id):
        """Get the polarity of other opinions.

        Args:
            term_to_position_list: dict
            term_id: str
        Returns:
            other_opinions: set
            other_polarity: set
        """
        other_polarity = set()
        other_opinions = set()
        for other_term_id in term_to_position_list:
            if other_term_id != term_id:
                other_polarity.add(term_to_position_list[other_term_id]['polarity'])
                for other_opi in term_to_position_list[other_term_id]['opinions']:
                    other_opinions.add(other_opi[0])
        return other_opinions, other_polarity

    def trans_aspect_polarity(self, aspect_term, words_list):
        """Reverse the polarity of the aspect.

        Args:
            aspect_term: dict
            words_list: str
        Returns:
            trans_words: list
            trans_polarity: str
        """
        aspect_polarity = aspect_term['polarity']
        opinions_position = aspect_term['opinions']
        if aspect_polarity == 'positive':
            trans_words, trans_opinions = self.reverse(words_list, opinions_position)
            trans_polarity = 'negative'
        elif aspect_polarity == 'negative':
            trans_words, trans_opinions = self.reverse(words_list, opinions_position)
            trans_polarity = 'positive'
        else:
            trans_words1, trans_opinions1 = self.reverse(words_list, opinions_position)
            trans_words2, trans_opinions2 = self.reverse(words_list, opinions_position)
            trans_words = random.choice([trans_words1, trans_words2])
            trans_polarity = 'neutral'
        return trans_words, trans_polarity

    def trans_conjunction(self, term_list, term_id, trans_words, trans_polarity):
        """Transform the conjunction words in sentence.

        Args:
            term_list: dict
            term_id: str
            trans_words: list
            trans_polarity: str
        Returns:
            trans_words: list
        """
        aspect_opinions = set()
        conjunction_list = ['and', 'but']
        aspect_opinions.add(aspect_opi[0] for aspect_opi in term_list[term_id]['opinions'])
        other_opinions, other_polarity = self.get_other_opinions(term_list, term_id)
        if len(other_polarity) > 0 and len(aspect_opinions & other_opinions) == 0:
            conjunction_idx = self.get_conjunction_idx(trans_words, term_list[term_id], conjunction_list)
            if conjunction_idx is not None and trans_polarity not in other_polarity:
                trans_words[conjunction_idx] = 'but'
            elif conjunction_idx is not None and trans_polarity in other_polarity:
                trans_words[conjunction_idx] = 'and'
        return trans_words


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
    reverse_target = AbsaReverseTarget()
    trans = reverse_target.transform(data_sample)
    print(trans)
