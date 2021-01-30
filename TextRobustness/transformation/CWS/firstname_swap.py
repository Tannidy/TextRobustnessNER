from TextRobustness.transformation import Transformation
import random
from TextRobustness.component.sample import CWSSample
from TextRobustness.common.settings import NAME_PATH, WORD_LIST_PATH
from TextRobustness.common.utils.load import cws_get_list


class FirstNameSwap(Transformation):
    """ Make the first word of the surname and the preceding word form a word,
                and the last word of the name and the following word form a word
    Attributes:
        firstname_list family name dictionary
        word_list dictionary of words
        word_end_dict
        name_dict A dictionary ending with a surname

    """

    def __init__(self, **kwargs):
        super().__init__()
        self.firstname_list = cws_get_list(NAME_PATH)
        self.word_list = cws_get_list(WORD_LIST_PATH)
        self.word_end_dict, self.name_dict = self.make_dict()

    def make_dict(self):
        """

            Returns:Last name dictionary and first name dictionary

        """
        word_end_dict = {}
        name_dict = {}
        for word in self.word_list:
            if len(word) > 1:
                if word[1:] not in word_end_dict:
                    word_end_dict[word[1:]] = [word[0]]
                elif word[0] not in word_end_dict[word[1:]]:
                    word_end_dict[word[1:]] += [word[0]]
                if word[-1:] in self.firstname_list:
                    if word[:-1] not in name_dict:
                        name_dict[word[:-1]] = [word[-1:]]
                    elif word[-1:] not in name_dict[word[:-1]]:
                        name_dict[word[:-1]].append(word[-1])

        return word_end_dict, name_dict

    def _transform(self, data_sample, n=5, **kwargs):
        """ We randomly generated five sets of data.

        Args:
            data_sample:data_sample the data which need be changed
            n: number of generated data
            **kwargs:

        Returns:
            trans_sample a list of sample

        """
        assert isinstance(data_sample.y.field_value, list)
        # get sentence label and ner tag
        origin_sentence = data_sample.x.field_value
        origin_label = data_sample.y.field_value
        ner_label, _ = data_sample.x.ner()
        trans_sample = []

        for i in range(n):
            # change function
            change_pos, change_list = self._get_transformations(origin_sentence, origin_label, ner_label)
            if len(change_pos) == 0:
                return [sample]
            x = data_sample.x.replace_at_ranges(change_pos, change_list)
            trans_sample.append(data_sample.update(x, data_sample.y))

        return trans_sample

    def _get_transformations(self, sentence, label, ner_label):
        """

        Args:
            sentence: chinese sentence
            label: Chinese word segmentation tag
            ner_label: sentence's ner tag

        Returns:
            change_pos, change_list
                two list include the pos which changed and the label which changed

        """
        assert len(sentence) == len(label)

        change_pos = []
        change_list = []
        if len(ner_label):
            for ner in ner_label:
                tag, start, end = ner
                # Determine whether it is a name based on the ner tag and the word segmentation tag
                if tag != 'Nh' or label[start] != 'B' \
                        or label[end] != 'E' \
                        or label[start + 1:end] != ['M'] * (end - start - 1):
                    continue
                # Combine the last name and the previous n words into a word, and get a list of replacement words
                s = ''
                change = []
                for i in range(1, 6):
                    if start < i:
                        break
                    s = sentence[start - i] + s
                    if s in self.name_dict:
                        change += self.name_dict[s]
                if len(change) > 0:
                    change_pos += [start]
                    change_list += [random.choice(change)]
                # The name and the following n letters form a word, and get a list of replacement words
                s = ''
                change = []
                for j in range(1, 5):
                    if end + j >= len(label):
                        break
                    s += sentence[end + j]
                    if s in self.word_end_dict:
                        change += self.word_end_dict[s]
                if len(change) > 0:
                    change_pos += [end]
                    change_list += [random.choice(change)]
        return change_pos, change_list


if __name__ == "__main__":
    sent1 = '我朝小明招了招手'
    data_sample = CWSSample({'x': sent1, 'y': ['S', 'S', 'B', 'E', 'B', 'M', 'E', 'S']})
    swap_ins = FirstNameSwap()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
