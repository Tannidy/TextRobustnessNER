"""
Reverse the polarity of non-target in ABSA task
==========================================================
"""

from TextRobustness.component.sample import ABSASample
from TextRobustness.transformation.ABSA.absa_transformation import ABSATransformation


class AbsaReverseNonTarget(ABSATransformation):
    """ Transforms the polarity of non-target by replacing its opinion words
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
            trans_words = self.trans_other_polarity(term_list, term_id, words_list)
            trans_sentence = self.get_sentence(trans_words, sentence)
            aspect_from, aspect_to = self.get_term_span(trans_sentence, term)

            trans_sample = {
                'x': trans_sentence,
                'y': [term_list[term_id]['polarity']],
                'term_list': {
                    'term': term,
                    'id': term_id,
                    'from': aspect_from,
                    'to': aspect_to,
                    }
                }
            trans_sample = ABSASample(trans_sample)
            trans_samples.append(trans_sample.dump())
        return trans_samples

    def trans_other_polarity(self, term_list, term_id, words_list):
        """Transform the polarity of other opinions.

        Args:
            term_list: dict
            term_id: str
            words_list: list
        Returns:
            trans_words: list
        """
        aspect_term = term_list[term_id]
        other_id_list = [idx for idx in term_list]
        other_id_list.remove(term_id)
        trans_words = words_list
        trans_opinion_positions = []

        for other_index, other_id in enumerate(other_id_list):
            other_term = term_list[other_id]
            non_overlap_opinion = []
            other_opinion = other_term['opinions']
            for i in other_opinion:
                if i in trans_opinion_positions:
                    continue
                else:
                    non_overlap_opinion.append(i)
            if len(non_overlap_opinion) == 0:
                continue
            old_trans_words = trans_words
            trans_words, trans_position = self.trans_term_polarity(
                aspect_term, other_term, trans_words, non_overlap_opinion)
            if trans_words != old_trans_words:
                trans_opinion_positions.append(i for i in trans_position)

        return trans_words

    def trans_term_polarity(self, aspect_term, other_term, trans_words, non_overlap_opinion):
        """Transform the polarity of a certain term.

        Args:
            aspect_term: dict
            other_term: dict
            trans_words: list
            non_overlap_opinion list:
        Returns:
            trans_words: list
            trans_position: list
        """
        aspect_polarity = aspect_term['polarity']
        term_polarity = other_term['polarity']
        if aspect_polarity == term_polarity in ['positive', 'negative']:
            trans_words, trans_position = self.reverse(trans_words, non_overlap_opinion)
            trans_words = self.trans_conjunction(aspect_term, trans_words)
        else:
            trans_words, trans_position = self.exaggerate(trans_words, non_overlap_opinion)
        return trans_words, trans_position

    def trans_conjunction(self, aspect_term, trans_words):
        """Transform the conjunction words in sentence.

        Args:
            aspect_term: dict
            trans_words: list
        Returns:
            trans_words: list
        """
        conjunction_list = ['and']
        conjunction_idx = self.get_conjunction_idx(trans_words, aspect_term, conjunction_list)
        if conjunction_idx is not None:
            trans_words[conjunction_idx] = 'but'
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
    reverse_target = AbsaReverseNonTarget()
    trans = reverse_target.transform(data_sample)
    print(trans)
