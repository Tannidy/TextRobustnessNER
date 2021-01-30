"""
contract sentence by common abbreviations in Twitter
==========================================================
"""
import re
import random
import string
import numpy as np
from TextRobustness.component.sample import SASample
from TextRobustness.transformation import Transformation
from TextRobustness.common.settings import TWITTER_PATH
from TextRobustness.common.utils.load import load_twitter_words

twitter_dic = load_twitter_words(TWITTER_PATH)


class WordSwapTwitter(Transformation):
    """
    Transforms input by common abbreviations in Twitter
    """

    def __init__(self, random_type='at', processor=None):
        super().__init__(processor=processor)
        self.random_type = random_type
        if random_type and random_type not in ['at', 'url', 'random']:
            raise ValueError(f"random_type value not one of ['at', 'url', 'random', None]")

    def _transform(self, sample, transform_field='x', **kwargs):
        """ Transform text string according transform_field.

        Args:
            sample: dict
                input data, normally one data component.
            transform_fields: str or list
                indicate which fields to transform,
                for multi fields , substitute them at the same time.
            n: int
                number of generated samples
        Returns:
            list
                transformed sample list.
        """
        transform_fields = [transform_field] if isinstance(transform_field, str) else transform_field

        trans_samples = []

        for i, field in enumerate(transform_fields):
            tokens = sample.get_words(field)
            get_contract_sample = self._get_contract_sample(tokens, field, sample)
            if not get_contract_sample:
                return []
            else:
                random_text = ''
                if self.random_type == 'at':
                    random_text = self.random_at(random.randint(1, 10))
                if self.random_type == 'url':
                    random_text = self.random_url(random.randint(1, 5))
                if self.random_type == 'random':
                    random_at_text = self.random_at(random.randint(1, 10))
                    random_url_text = self.random_url(random.randint(1, 5))
                    random_text = random.choice([random_at_text, random_url_text])
                if random_text:
                    random_index = random.choice([0, len(get_contract_sample.get_words(field)) - 1])
                    random_sample = get_contract_sample.insert_field_before_index(field, random_index, random_text)
                    trans_samples.append(random_sample)
                else:
                    trans_samples.append(get_contract_sample)

            return trans_samples

    @staticmethod
    def _get_contract_sample(tokens, field, sample):
        contract_sample = sample
        sentence = ' '.join(token for token in tokens)
        contract_list = []
        indexes_list = []
        for twitter_words in twitter_dic:
            if twitter_words in sentence:
                len_twitter_words = len(twitter_words.split(' '))
                len_sent = sentence.find(twitter_words)
                contract_word = [twitter_dic[twitter_words]]
                len_contract_word = len(contract_word)
                start_index = len(sentence[:len_sent].split(' ')) - 1
                index_list = []
                for i in range(len_twitter_words):
                    index_list.append(start_index + i)
                indexes_list.append(index_list)
                if len_twitter_words != len_contract_word:
                    for i in range(len_twitter_words - len_contract_word):
                        contract_word.append(' ')

                contract_list.append(contract_word)

        if indexes_list:
            for i, indexes in enumerate(indexes_list):
                if len(indexes) == len(contract_list[i]):
                    contract_sample = contract_sample.replace_field_at_indices(field, indexes, contract_list[i])

            new_tokens = contract_sample.get_words(field)
            space_list = []
            for i in range(len(new_tokens)):
                if new_tokens[i] == ' ':
                    space_list.insert(0, i)

            for i in space_list:
                contract_sample = contract_sample.delete_field_at_index(field, i)

            return contract_sample

    @staticmethod
    def random_string(n):
        return ''.join(np.random.choice([x for x in string.ascii_letters + string.digits], n))

    def random_url(self, n=2):
        return 'https://{0}.{1}/{2}'.format(self.random_string(n), self.random_string(n), self.random_string(n))

    def random_at(self, n=5):
        return '@{0}'.format(self.random_string(n))


if __name__ == "__main__":
    sent1 = "face to face, the details, bye for now, cool, email, more to follow."
    data_sample = SASample({'x': sent1, 'y': "negative"})
    swap_ins = WordSwapTwitter()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
