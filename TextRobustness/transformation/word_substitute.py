"""
WordSubstitute Base Class
============================================

"""

import random
import string
from abc import abstractmethod

from TextRobustness.transformation import Transformation
from TextRobustness.common.settings import STOP_WORDS, ORIGIN
from TextRobustness.common.utils.list_op import trade_off_sub_words


class WordSubstitute(Transformation):
    """ Word replace transformation to implement normal word replace functions.

    Attributes:
        trans_min: int
            Minimum number of word will be augmented.
        trans_max: int
            Maximum number of word will be augmented. If None is passed, number of augmentation is
            calculated via aup_char_p. If calculated result from aug_p is smaller than aug_max, will use calculated
            result from aup_char_p. Otherwise, using aug_max.
        trans_p: float
            Percentage of word will be augmented.
        stop_words: list
            List of words which will be skipped from augment operation.
        processor: TextRobustness.common.preprocess.TextProcessor.
        get_pos: bool
            whether pass pos tag to _get_substitute_words API.

    """
    def __init__(self, trans_min=1, trans_max=10, trans_p=0.1, stop_words=None, **kwargs):
        super().__init__()
        self.trans_min = trans_min
        self.trans_max = trans_max
        self.trans_p = trans_p
        self.stop_words = STOP_WORDS if not stop_words else stop_words
        # set this value to avoid meaningless pos tagging
        self.get_pos = False

    def _transform(self, sample, field='x', n=1, **kwargs):
        """ Transform text string according field.

        Args:
            sample: dict
                input data, normally one data component.
            fields: str
                indicate which field to apply transformation
            n: int
                number of generated samples
        Returns:
            transformed sample list.

        """
        tokens = sample.get_words(field)
        tokens_mask = sample.get_mask(field)

        # return up to (len(sub_indices) * n) candidates
        pos_info = sample.get_pos(field) if self.get_pos else None
        legal_indices = self.skip_aug(tokens, tokens_mask, pos=pos_info)

        if not legal_indices:
            return []

        sub_words, sub_indices = self._get_substitute_words(tokens, legal_indices, pos=pos_info, n=n)
        # select property candidates
        trans_num = self.get_trans_cnt(len(tokens))
        sub_words, sub_indices = trade_off_sub_words(sub_words, sub_indices, trans_num)

        if not sub_words:
            return []

        trans_samples = []

        for i in range(len(sub_words[0])):
            single_sub_words = [sub_word[i] for sub_word in sub_words]
            trans_samples.append(sample.replace_field_at_indices(field, sub_indices, single_sub_words))

        return trans_samples

    def _get_substitute_words(self, words, legal_indices, pos=None, n=5):
        """ Returns a list containing all possible words .

        Args:
            words: all words
            legal_indices: indices which has not been skipped
            pos: None or list of pos tags
            n: max candidates for each word to be substituted

        Returns:
            list of list
        """
        # process each legal words to get maximum transformed samples
        legal_words = [words[index] for index in legal_indices]
        legal_words_pos = [pos[index] for index in legal_indices] if self.get_pos else None

        candidates_list = []
        candidates_indices = []

        for index, word in enumerate(legal_words):
            _pos = legal_words_pos[index] if self.get_pos else None
            candidates = self._get_candidates(word, pos=_pos, n=n)
            # filter no word without candidates
            if candidates:
                candidates_indices.append(legal_indices[index])
                candidates_list.append(self._get_candidates(word, pos=_pos, n=n))

        return candidates_list, candidates_indices

    @abstractmethod
    def _get_candidates(self, word, pos=None, n=5):
        """ Returns a list containing all possible words .

        Args:
            word: str
            pos: str

        Returns:
            candidates list
        """
        raise NotImplementedError

    @abstractmethod
    def skip_aug(self, tokens, mask, pos=None):
        """ Returns the index of the replaced tokens.

        Args:
            tokens: list
                tokenized words or word with pos tag pairs

        Returns:
            list
        """
        raise NotImplementedError

    def is_stop_words(self, token):
        """ Judge whether the input word belongs to the stop words vocab.

        Args:
            token: str.

        Returns:
            bool
        """
        return self.stop_words is not None and token in self.stop_words

    def pre_skip_aug(self, tokens, mask):
        """ Skip the tokens in stop words list or punctuation list.

        Args:
            tokens: list
            mask: list
                Indicates whether each word is allowed to be substituted.
                ORIGIN is allowed, while TASK_MASK and MODIFIED_MASK is not.

        Returns:
            list.
                List of possible substituted token index.
        """
        assert len(tokens) == len(mask)
        results = []

        for token_idx, token in enumerate(tokens):
            # skip punctuation
            if token in string.punctuation:
                continue
            # skip stopwords by list
            if self.is_stop_words(token):
                continue
            if mask[token_idx] != ORIGIN:
                continue

            results.append(token_idx)

        return results

    @classmethod
    def sample_num(cls, x, num):
        """ Get 'num' samples from x.

        Args:
            x: list to sample
            num: sample number

        Returns:
            max 'num' unique samples.

        """
        if isinstance(x, list):
            num = min(num, len(x))
            return random.sample(x, num)
        elif isinstance(x, int):
            num = min(num, x)
            return random.sample(range(0, x), num)

    def get_trans_cnt(self, size):
        """ Get the num of words/chars transformation.

        Args:
            size: the size of target sentence

        Returns:
            number of words to apply transformation.

        """

        cnt = int(self.trans_p * size)

        if cnt < self.trans_min:
            return self.trans_min
        if self.trans_max is not None and cnt > self.trans_max:
            return self.trans_max

        return cnt

    @staticmethod
    def token2chars(word):
        return list(word)

    @staticmethod
    def chars2token(chars):
        assert isinstance(chars, list)

        return ''.join(chars)
