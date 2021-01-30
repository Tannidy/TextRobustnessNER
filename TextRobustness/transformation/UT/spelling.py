"""
Transformation that apply spelling error simulation to textual input.
==========================================================

"""


from TextRobustness.common.settings import SPELLING_ERROR_DIC
from TextRobustness.transformation import WordSubstitute


class SpellingErrorRules:
    def __init__(self, dict_path, include_reverse=True):
        self.dict_path = dict_path
        self.include_reverse = include_reverse

        self._init()

    def _init(self):
        self.rules = {}
        self.read(self.dict_path)

    def read(self, model_path):
        with open(model_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                tokens = line.split(' ')
                # Last token include newline separator
                tokens[-1] = tokens[-1].replace('\n', '')

                key = tokens[0]
                values = tokens[1:]

                if key not in self.rules:
                    self.rules[key] = []

                self.rules[key].extend(values)
                # Remove duplicate mapping
                self.rules[key] = list(set(self.rules[key]))
                # Build reverse mapping
                if self.include_reverse:
                    for value in values:
                        if value not in self.rules:
                            self.rules[value] = []
                        if key not in self.rules[value]:
                            self.rules[value].append(key)

    def predict(self, data):
        if data not in self.rules:
            return None

        return self.rules[data]


class Spelling(WordSubstitute):
    """ Transformation that leverage pre-defined spelling mistake dictionary to simulate spelling mistake.
        
        https://arxiv.org/ftp/arxiv/papers/1812/1812.04718.pdf

        Attributes:
            trans_min: int
                Minimum number of character will be augmented.
            trans_max: int
                Maximum number of character will be augmented. If None is passed, number of augmentation is
                calculated via aup_char_p. If calculated result from aug_p is smaller than aug_max, will use calculated
                result from aup_char_p. Otherwise, using aug_max.
            trans_p: float
                Percentage of character (per token) will be augmented.
            stop_words: list
                List of words which will be skipped from augment operation.
            processor: TextRobustness.common.preprocess.TextProcessor.
            include_reverse: bool.
                whether build reverse map according to spelling error list.

    """
    def __init__(self, trans_min=1, trans_max=10, trans_p=0.3, stop_words=None,
                 include_reverse=True, rules_path=None, **kwargs):
        super().__init__(trans_min=trans_min, trans_max=trans_max, trans_p=trans_p,
                         stop_words=stop_words)

        self.rules_path = rules_path if rules_path else SPELLING_ERROR_DIC
        self.include_reverse = include_reverse
        self.rules = self.get_rules()

    def __repr__(self):
        return 'Spelling'

    def skip_aug(self, tokens, mask, **kwargs):
        pre_skipped_idxes = self.pre_skip_aug(tokens, mask)
        results = []

        for token_idx in pre_skipped_idxes:
            # Some words do not exit. It will be excluded in lucky draw.
            token = tokens[token_idx]
            if token in self.rules.rules and len(self.rules.rules[token]) > 0:
                results.append(token_idx)

        return results

    def _get_candidates(self, word, n=1, **kwargs):
        """ Get a list of transformed tokens.

            default one word replace one char.

        Args:
            word: str
                token word to transform.
            n: int
                number of transformed tokens to generate.

        Returns:
            list
        """
        return self.sample_num(self.rules.predict(word), n)

    def get_rules(self):
        return SpellingErrorRules(self.rules_path, self.include_reverse)


if __name__ == "__main__":
    from TextRobustness.component.sample import SASample

    sent1 = 'The quick brown fox jumps over the lazy dog.'
    data_sample = SASample({'x': sent1, 'y': "negative"})
    trans = Spelling()
    x = trans.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
