import random
from TextRobustness.transformation import Transformation
from TextRobustness.component.sample import CWSSample
from TextRobustness.common.settings import NUM_LIST, NUM_FLAG1, NUM_FLAG2, NUM_BEGIN, NUM_END


class NumberSwap(Transformation):
    """ Make short numbers grow into long numbers.

    Attributes:
        num_list: the list which include all the number we need
            if you want to change it you must change NUM_LIST, NUM_FLAG1, NUM_FLAG2, NUM_BEGIN, NUM_END
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.num_list = NUM_LIST

    def _transform(self, sample, n=5, **kwargs):
        """We randomly generated five sets of data.

        Args:
            sample: sample the data which need be changed
            n: number of generated data
            **kwargs:

        Returns:trans_sample a list of sample

        """
        assert isinstance(sample.y.field_value, list)
        # get sentence and label
        origin_sentence = sample.x.field_value
        origin_label = sample.y.field_value
        trans_sample = []

        for i in range(n):
            # change function
            change_pos, change_sentence, change_label = self._get_transformations(origin_sentence, origin_label)
            if len(change_pos) == 0:
                return [sample]
            x = sample.x.replace_at_ranges(change_pos, change_sentence)
            y = sample.y.replace_at_ranges(change_pos, change_label)
            trans_sample.append(sample.update(x, y))

        return trans_sample

    def _get_transformations(self, sentence, label):
        """Number change function.

        Args:
            sentence: chinese sentence
            label: Chinese word segmentation tag

        Returns:
            change_pos, change_sentence, change_labels
                three list include the pos which changed the word which changed and the label which changed

        """
        assert len(sentence) == len(label)

        start = 0
        change_pos = []
        change_sentence = []
        change_labels = []

        while start < len(sentence):
            # find the number
            if sentence[start] in self.num_list:
                if label[start] == 'S' and sentence[start - 1:start] != '第' and sentence[start] != '一':
                    if self.num_list.index(sentence[start]) < 10:
                        # if single number
                        # create a 至 b
                        ra = random.randint(1, 10)
                        rb = random.randint(1, 10)

                        while rb == ra:
                            rb = random.randint(1, 10)
                        if ra > rb:
                            tmp = ra
                            ra = rb
                            rb = tmp
                        change = self.num_list[ra] + '至' + self.num_list[rb]
                        change_label = ['B', 'M', 'E']
                        change_pos.append([start, start + 1])
                        change_sentence.append(change)
                        change_labels.append(change_label)
                    else:
                        # not a single number
                        change, change_label = self.number_change(sentence, label, start, start)
                        if change != '':
                            change_pos.append([start, start + 1])
                            change_sentence.append(change)
                            change_labels.append(change_label)
                elif label[start] == 'B':
                    # Process numbers with length greater than 1
                    flag = 1
                    end = start
                    while label[end] != 'E':
                        end += 1
                        if sentence[end] not in self.num_list:
                            flag = 0
                            break
                    if flag:
                        change, change_label = self.number_change(sentence, label, start, end)
                        change_pos.append([start, end + 1])
                        change_sentence.append(change)
                        change_labels.append(change_label)
                    start = end + 1
            start += 1

        return change_pos, change_sentence, change_labels

    def create_num(self, pos):
        # create chinese number
        if pos <= NUM_FLAG1:
            return str(self.num_list[random.randint(NUM_BEGIN, NUM_END)])

        res = ''
        if pos <= NUM_FLAG2:
            res += self.num_list[random.randint(NUM_BEGIN, NUM_END)] + self.num_list[pos] + self.create_num(pos - 1)
            return res

        res += self.create_num(pos - 1) + self.num_list[pos] + self.create_num(pos - 1)
        return res

    def number_change(self, sentence, label, start, end):
        assert len(label) == len(sentence)
        # Digital conversion of start to end
        max_num = 0
        change = ''
        change_label = []

        for i in range(start, end + 1):
            max_num = max(self.num_list.index(sentence[i]), max_num)
        if end - start > 1 and max_num < 10:
            # Special numbers are not deformed
            return change, label
        change = self.create_num(max_num)
        seed = random.randint(0, 2)

        if len(change) == 1:
            seed = 0
        if seed == 1:
            change = change[:-1] + random.choice(['来', '余'])
        elif seed == 0:
            ca = self.create_num(max_num)
            cb = self.create_num(max_num)
            while ca == cb:
                cb = self.create_num(max_num)
            if self.compare(ca, cb):
                tmp = ca
                ca = cb
                cb = tmp
            change = ca + random.choice(['至', '到']) + cb
        if len(change) > 1:
            change_label = ['B'] + ['M'] * (len(change) - 2) + ['E']

        return change, change_label

    def compare(self, num1, num2):
        # compare two number
        for i in range(len(num1)):
            if num1[i] == num2[i]:
                continue
            return self.num_list.index(num1[i]) > self.num_list.index(num2[i])
        return True


if __name__ == "__main__":
    sent1 = '九百'
    data_sample = CWSSample({'x': sent1, 'y': ['B', 'E']})
    swap_ins = NumberSwap()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
