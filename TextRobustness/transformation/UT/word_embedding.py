"""
Swapping words by Glove
==========================================================
"""
from TextRobustness.common.settings import EMBEDDING_PATH
from TextRobustness.component.sample import SASample
from TextRobustness.transformation import WordSubstitute
from TextRobustness.common.utils.load import load_embedding_words

sim_dic = load_embedding_words(EMBEDDING_PATH)


# TODO
class WordEmbedding(WordSubstitute):
    """ Transforms an input by replacing its words by Glove.
        Offline Vocabulary is provided.
    """

    def __init__(self, trans_min=1, trans_max=10, trans_p=0.1,
                 stop_words=None):
        super().__init__(trans_min=trans_min, trans_max=trans_max, trans_p=trans_p,
                         stop_words=stop_words)
        self.get_pos = True

    def __repr__(self):
        return 'WordEmbedding'

    def _get_candidates(self, word_pos, n=5):
        """ Returns a list containing all possible words with 1 character
            replaced by word embedding.
        """
        word, pos = word_pos
        synonym = set()
        sim_list = self.word_in_sim_dic(word)

        for syn_word in sim_list:
            # skip origin word
            if syn_word == word:
                continue
            # skip synonym word with different pos tag
            if self.processor.filter_candidates_by_pos(word_pos, [syn_word]):
                synonym.add(syn_word)

        return list(synonym)[:n]

    @staticmethod
    def word_in_sim_dic(word):
        if word in sim_dic:
            return sim_dic[word]
        else:
            return []

    def skip_aug(self, tokens_info, mask):
        tokens = [token_info[0] for token_info in tokens_info]
        return self.pre_skip_aug(tokens, mask)


if __name__ == "__main__":
    sent1 = 'The quick brown fox jumps over the lazy dog .'
    data_sample = SASample({'x': sent1, 'y': "negative"})
    swap_ins = WordEmbedding()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
