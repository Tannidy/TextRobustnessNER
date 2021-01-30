"""
Swap/delete/add random character for entities
==========================================================

"""

from TextRobustness.common.utils.word_op import *
from TextRobustness.component.sample.ner_sample import NerSample
from TextRobustness.transformation import Transformation


class EntityTyposSwap(Transformation):
    """ Transformation that simulate typos error to transform sentence.

        https://arxiv.org/pdf/1711.02173.pdf

    Attributes:

        processor: TextRobustness.common.preprocess.TextProcessor.
        mode : str.
            just support ['random', 'replace', 'swap', 'insert', 'delete'].
        skip_first_char: bool.
            whether skip the first char of target word.
        skip_last_char: bool.
            whether skip the last char of target word.

    """
    def __init__(self, mode="random", skip_first_char=False, skip_last_char=False, **kwargs):
        super().__init__()
        self._mode = mode
        self.skip_first_char = skip_first_char
        self.skip_last_char = skip_last_char

    # TODO， n实现
    def _transform(self, sample, n=1, **kwargs):
        """Transform data sample to a list of Sample.
        Args:
            input_sample: NerSample
                Data sample for augmentation.
            n: int
                Default is 5. MAx number of unique augmented output.
            **kwargs:

        Returns: Augmented data

        """
        new_sample_list = []
        entity_in_seq = sample.get_value('entities')

        cand_entities = []

        for entity in entity_in_seq:
            cur_entity = entity['entity']
            entity_tokens = cur_entity.split(" ")
            rep_idx = random.randint(0, len(entity_tokens)-1)
            rep_tokens = self._get_replacement_words(entity_tokens[rep_idx], n=n)

            rep_entities = [entity_tokens[:rep_idx] + [rep_token] + entity_tokens[rep_idx+1:]
                            for rep_token in rep_tokens]

            new_sample_list.append(sample.entity_replace(entity['start'],
                                                         entity['end'],
                                                         " ".join(new_entity_tokens),
                                                         entity['tag']))
        return new_sample_list

    def _get_replacement_words(self, word, n=1):
        """ Returns a list of words with typo errors.

        Args:
            word: str
            n: int
                number of try times

        Returns:
            list
        """

        candidates = set()

        for i in range(n):
            typo_method = self._get_typo_method()
            candidates.add(typo_method(word, num=n))

        return list(candidates)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode_value):
        assert mode_value in ['random', 'replace', 'swap', 'insert', 'delete']
        self._mode = mode_value

    def _get_typo_method(self):
        if self._mode == 'replace':
            return replace
        elif self._mode == 'swap':
            return swap
        elif self._mode == 'insert':
            return insert
        elif self._mode == 'delete':
            return delete
        else:
            return random.choice([replace, swap, insert, delete])


if __name__ == "__main__":
    sent1 = 'EU rejects German call to boycott British lamb .'
    data_sample = NerSample({'x': sent1, 'y': ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']})
    swap_ins = EntityTyposSwap()
    x = swap_ins.transform(data_sample, n=5)

    for ner_sample in x:
        print(ner_sample.dump())
    # sent1 = ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
    # data_sample = {'x': sent1, 'y': ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']}
    # swap_ins = EntityTyposSwap()
    # for i in range(5):
    #     print(swap_ins._transform(data_sample))
