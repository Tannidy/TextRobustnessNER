from TextRobustness.transformation import Transformation
from ltp import LTP
import torch
from transformers import BertTokenizer, BertForMaskedLM
from TextRobustness.component.sample import CWSSample


class GenerateWord(Transformation):
    """ Use Bert to generate words.

    """
    def __init__(self, **kwargs):
        super().__init__()

    def _transform(self, sample, **kwargs):
        """ In this function, because there is only one deformation mode, only one set of outputs is output.

        Args:
            sample: sample the data which need be changed
            **kwargs:

        Returns:
            trans_sample a list of sample

        """
        assert isinstance(sample.y.field_value, list)
        # get sentence label and pos tag
        origin_sentence = sample.x.field_value
        origin_label = sample.y.field_value
        pos_tags = sample.x.pos_tag()
        transform_sentence, transform_label = self._get_transformations(origin_sentence, origin_label, pos_tags)
        return [CWSSample({'x': transform_sentence, 'y': transform_label})]

    def create_word(self, sentence):
        """

        Args:
            sentence: the sentence with [MASK]

        Returns:
            the change sentence

        """
        # Crete the word we need
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        text = '[CLS] ' + sentence
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # Create the segments tensors.
        segments_ids = [0] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Load pre-trained model (weights)
        model = BertForMaskedLM.from_pretrained('bert-base-chinese')
        model.eval()
        masked_index = tokenized_text.index('[MASK]')
        masked_index1 = masked_index + 1
        # Predict all tokens
        with torch.no_grad():
            predictions = model(tokens_tensor, segments_tensors)

        predicted_index = torch.argmax(predictions[0][0][masked_index]).item()
        predicted_index1 = torch.argmax(predictions[0][0][masked_index1]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        predicted_token1 = tokenizer.convert_ids_to_tokens([predicted_index1])[0]

        text = text.strip().split(' ')
        text = text[1:]
        sentence = ''
        start = 0
        # Determine whether the generated words meet the requirements
        if len(predicted_token) != 1 or len(predicted_token) != 1 or self.is_word(predicted_token + predicted_token1):
            return ''
        # Change the generated sentence
        while start < len(text):
            if text[start] == '[MASK]':
                sentence += predicted_token + predicted_token1
                start += 1
            else:
                sentence += text[start]
            start += 1
        return sentence

    def _get_transformations(self, sentence, label, pos_tags):
        """Generate word function.

        Args:
            sentence: chinese sentence
            label: Chinese word segmentation tag
            pos_tags: sentence's pos tag

        Returns:
            sentence, label
                two list include the pos which changed and the label which changed

        """
        assert len(sentence) == len(label)
        cnt = 0

        for i in range(len(pos_tags)):
            tag, start, end = pos_tags[i]
            start += cnt
            end += cnt
            # find the pos that can generate word
            # Situation 1: v + single n
            # we generate double n replace single n
            if label[start] == 'B' and label[start + 1] == 'E' and \
                    i < len(pos_tags) - 1 and pos_tags[i][0] == 'v' \
                    and pos_tags[i + 1][0] == 'n' and pos_tags[i + 1][2] + cnt == start + 1:
                token = ''
                for j in range(len(sentence)):
                    if j != start + 1:
                        token += sentence[j] + ' '
                    else:
                        token += '[MASK] [MASK] '
                change = self.create_word(token)
                if change != '':
                    sentence = self.create_word(token)
                    label = label[:start] + ['S', 'B', 'E'] + label[start + 2:]
                    cnt += 1
                    start += 1
            # Situation 1: n + n + n
            # we generate double n replace single n and split one word into two
            elif label[start:start + 3] == ['B', 'M', 'E'] and \
                    tag == 'n' and end - start == 2:
                token = ''
                start += 2
                for i in range(len(sentence)):
                    if i != start:
                        token += sentence[i] + ' '
                    else:
                        token += '[MASK] [MASK] '
                change = self.create_word(token)
                if change != '':
                    sentence = change
                    label = label[:start - 1] + ['E', 'B', 'E'] + label[start + 1:]
                    cnt += 1
                    start += 1
            start += 1

        return sentence, label

    @staticmethod
    def is_word(sentence):
        """ Judge whether it is a word.

        Args:
            sentence: input sentence string

        Returns:
            bool
        """
        ltp = LTP()
        seg, hidden = ltp.seg([sentence])
        pos = ltp.pos(hidden)
        pos = pos[0]
        if len(pos) == 1 and pos[0] == 'n':
            return False
        return True


if __name__ == "__main__":
    sent1 = '玩具厂生产玩具'
    data_sample = CWSSample({'x': sent1, 'y': ['B', 'M', 'E', 'B', 'E', 'B', 'E']})
    swap_ins = GenerateWord()
    x = swap_ins.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
