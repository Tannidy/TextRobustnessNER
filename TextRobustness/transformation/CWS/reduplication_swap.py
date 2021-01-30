from TextRobustness.transformation import Transformation
from TextRobustness.component.sample import CWSSample
from TextRobustness.common.settings import AABB_PATH, ABAB_PATH, AONEA_PATH
from TextRobustness.common.utils.load import cws_get_list


class ReduplicationSwap(Transformation):
    """ Replace word with reduplication or a one a.

    Attributes:
        ABAB_list: word can be replaced by abab dictionary
        AABB_list: word can be replaced by aabb dictionary
        AoneA_list: word can be replaced by a one a dictionary

    """
    def __init__(self, **kwargs):
        super().__init__()
        self.ABAB_list = cws_get_list(ABAB_PATH)
        self.AABB_list = cws_get_list(AABB_PATH)
        self.AoneA_list = cws_get_list(AONEA_PATH)

    @staticmethod
    def get_pos_tag_list(pos_tag):
        # get the list of pos tag
        pos_list = []
        for i in pos_tag:
            tag, start, end = i
            pos_list += [tag] * (end - start + 1)
        return pos_list

    def _transform(self, data_sample, **kwargs):
        """

        Args:
            data_sample: the data which need be changed
            **kwargs:

        Returns:
            In this function, because there is only one deformation mode, only one set of outputs is output

        """
        assert isinstance(data_sample.y.field_value, list)
        # get sentence label and pos tag
        origin_sentence = data_sample.x.field_value
        origin_label = data_sample.y.field_value
        pos_tags = data_sample.x.pos_tag()
        pos_tags = self.get_pos_tag_list(pos_tags)
        trans_sample = []
        # change function
        change_pos, change_sentence, change_label = \
            self._get_transformations(origin_sentence, origin_label, pos_tags)
        if len(change_pos) == 0:
            return [sample]
        x = data_sample.x.replace_at_ranges(change_pos, change_sentence)
        y = data_sample.y.replace_at_ranges(change_pos, change_label)
        trans_sample.append(data_sample.update(x, y))

        return trans_sample

    def _get_transformations(self, sentence, label, pos_tag):
        """ ReduplicationSwap change function.

        Args:
            sentence: chinese sentence
            label: Chinese word segmentation tag
            pos_tag: sentence's pos tag

        Returns:
            change_pos, change_sentence, change_label
                three list include the pos which changed the word which changed and the label which changed

        """
        assert len(sentence) == len(label)

        change_pos = []
        change_sentence = []
        change_label = []
        start = 0
        while start < len(sentence) - 1:
            # find the word in AABB dictionary
            if label[start:start + 2] == ['B', 'E']\
                    and sentence[start] + sentence[start + 1] in self.AABB_list:
                # pos_tag[start:start + 2] == ['v', 'v'] 
                change_pos.append([start, start + 2])
                change_sentence.append(sentence[start] + sentence[start]
                                       + sentence[start + 1] + sentence[start + 1])
                change_label.append(['B', 'M', 'M', 'E'])
                start += 1
            # find ABAB word
            elif sentence[start] + sentence[start + 1] in self.ABAB_list and \
                    label[start:start + 2] == ['B', 'E']:
                # pos_tag[start:start + 2] == ['v', 'v']
                change_pos.append([start, start + 2])
                change_sentence.append(sentence[start] + sentence[start + 1]
                                       + sentence[start] + sentence[start + 1])
                change_label.append(['B', 'M', 'M', 'E'])
                start += 1
            elif pos_tag[start] == 'v' and pos_tag[start + 1] != 'v' and sentence[start] in self.AoneA_list:
                change_pos.append([start, start + 1])
                change_sentence.append(sentence[start] + '一'
                                       + sentence[start])
                if label[start] == 'S':
                    change_label.append(['B', 'M', 'E'])
                elif label[start] == 'E':
                    change_label.append(['M', 'M', 'E'])
                else:
                    change_label.append(['M', 'M', 'M'])
            start += 1

        return change_pos, change_sentence, change_label


if __name__ == "__main__":
    sent1 = '小明想朦胧'
    data_sample = CWSSample({'x': sent1, 'y': ['B', 'E', 'S', 'B', 'E']})
    swap_ins = ReduplicationSwap()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
