from TextRobustness.transformation import Transformation
from TextRobustness.component.sample import CWSSample
from TextRobustness.common.settings import abbreviation_path
from TextRobustness.common.utils.load import read_data


class AbbreviationChange(Transformation):
    """ Replace abbreviations with full names.

        Attributes:
            abbreviation_dict: the dictionary of abbreviation

    """
    def __init__(self, **kwargs):
        super().__init__()
        self.abbreviation_dict = self.make_dict(abbreviation_path)

    @staticmethod
    def make_dict(path):
        # read data
        dic = {}
        lines = read_data(path)
        for line in lines:
            line = line.strip().split(' ')
            dic[line[0]] = line[1:]
        return dic

    def _transform(self, data_sample, **kwargs):
        """

        Args:
            data_sample: the data which need be changed
            **kwargs:

        Returns:In this function, because there is only one deformation mode, only one set of outputs is output

        """
        assert isinstance(data_sample.y.field_value, list)
        # get sentence and label
        origin_sentence = data_sample.x.field_value
        origin_label = data_sample.y.field_value
        # change function
        change_pos, change_sentence, change_label = self._get_transformations(origin_sentence, origin_label)
        if len(change_pos) == 0:
            return [data_sample]
        x = data_sample.x.replace_at_ranges(change_pos, change_sentence)
        y = data_sample.y.replace_at_ranges(change_pos, change_label)

        return [data_sample.update(x, y)]

    def _get_transformations(self, sentence, label):
        """ Replace abbreviation function

        Args:
            sentence:  chinese sentence
            label: Chinese word segmentation tag

        Returns:change_pos, change_sentence, change_label
                three list include the pos which changed the word which changed and the label which changed

        """
        assert len(sentence) == len(label)
        start = 0
        change_pos = []
        change_sentence = []
        change_label = []

        while start < len(sentence):
            # find the abbreviation
            if label[start] == 'B':
                end = start + 1
                word = sentence[start]
                while label[end] != 'E':
                    word += sentence[end]
                    end += 1
                word += sentence[end]
                if word in self.abbreviation_dict:
                    # save abbreviations and change word segmentation labels
                    change_pos.append([start, end + 1])
                    change_sentence.append(self.abbreviation_dict[word])
                    change = []
                    for word in self.abbreviation_dict[word]:
                        if len(word) == 1:
                            change.append('S')
                        else:
                            change += ['B'] + ['M'] * (len(word) - 2) + ['E']
                    change_label.append(change)
                    start = end
            start += 1
        return change_pos, change_sentence, change_label


if __name__ == "__main__":
    sent1 = '央视'
    data_sample = CWSSample({'x': sent1, 'y': ['B', 'E']})
    swap_ins = AbbreviationChange()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
