"""
Transform all verb tenses in sentence
==========================================================
"""
import random
from TextRobustness.component.sample import SASample
from TextRobustness.transformation import SentenceSubstitute
from TextRobustness.common.settings import VERB_PATH
from TextRobustness.common.utils.load import load_verb_words

verb_dic = load_verb_words(VERB_PATH)
VERB_TAG = ['VB', 'VBP', 'VBZ', 'VBG', 'VBD', 'VBN']


# TODO
class Tense(SentenceSubstitute):
    """
    Transforms all verb tenses in sentence.
    Offline Vocabulary is provided.
    Notice: transformed sentence will have syntax errors.
    """

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'Tense'

    def _get_replace_sentences(self, tokens, n=5, **kwargs):
        replace_sentences = []
        transform_texts_list = self._get_transformation(tokens, n)
        if transform_texts_list:
            replace_sentences.append(transform_texts_list)

        return replace_sentences

    def _get_transformation(self, tokens, n):
        verb_location = self._get_verb_location(tokens)
        transformed_list = []
        if verb_location:
            transformed_texts_set = set()
            for i in range(n):
                transformed_texts = self._get_swap_tense_sentence(tokens, verb_location)
                for transformed_text in transformed_texts:
                    transformed_texts_set.add(transformed_text)
                transformed_list = list(transformed_texts_set)

        return transformed_list

    def _get_verb_location(self, tokens):
        sentence = ' '.join(token for token in tokens)
        #
        tag = self.processor.get_pos(sentence)
        # tag = pos_tag(tokens)
        verb_location = []
        for i, word_pos in enumerate(tag):

            if word_pos[1] in VERB_TAG:
                verb_location.append(i)

        return verb_location

    def _get_swap_tense_sentence(self, tokens, verb_location):
        for i in verb_location:
            word = tokens[i]
            candidate_list = self._get_tense_list(word)

            if candidate_list:
                tokens[i] = random.choice(candidate_list)

        return [' '.join(x for x in tokens)]

    @staticmethod
    def _get_tense_list(word):
        if word in verb_dic:
            return verb_dic[word]
        else:
            return []


if __name__ == "__main__":
    sent1 = 'The quick brown fox jumps over the lazy dog .'
    data_sample = SASample({'x': sent1, 'y': "negative"})
    swap_ins = Tense()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
