"""
Transforms an affirmative sentence into a negative sentence, or vice versa
==========================================================
"""
from TextRobustness.component.sample import SASample
from TextRobustness.transformation import Transformation


class SentReverseNeg(Transformation):
    """
    Transforms an affirmative sentence into a negative sentence, or vice versa
    """

    def __init__(self, processor=None, **kwargs):
        super().__init__(processor=processor)

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
            judge_sentence = self._judge_sentence(tokens)
            if judge_sentence == 'remove':
                del_sample = self._get_del_sample(tokens, field, sample)
                if del_sample:
                    trans_samples.append(del_sample)
            if judge_sentence == 'add':
                add_sample = self._get_add_sample(field, tokens, sample)
                if add_sample:
                    trans_samples.append(add_sample)

        return trans_samples

    @staticmethod
    def _judge_sentence(tokens):
        """

        Args:
            tokens:

        Returns:
            list: transformed_text or None

        """
        for i in tokens:
            if i in ['not', 'n\'t', 'don', 'didn', 'doesn', 'doesn', 'aren', 'isn', 'wasn', 'weren']:
                return 'remove'

        return 'add'

    @staticmethod
    def _check_sentence(tokens):
        """
        Check positive or negative

        """
        if len(tokens) < 3:
            return False
        if '?' in tokens:
            return False
        if tokens[0].lower() in ['are', 'is', 'be', 'am', 'was', 'were', 'how',
                                 'why', 'what', 'where', 'who', 'when', 'can',
                                 'do', 'did', 'does', 'could', 'should', 'would',
                                 'will', 'shall', 'thank', 'thanks']:
            return False
        else:
            return True

    def _parse_sentence(self, tokens):
        """
        Dependency Parsing
        """
        sentence = ' '.join(x for x in tokens)
        root_id_list = []
        parse_tokens = self.processor.get_dep_parser(sentence).split("\n")
        for i, token in enumerate(parse_tokens):
            token_pos = token.split("\t")
            if len(token_pos) < 4:
                continue
            if token_pos[3] in ['cop', 'root', 'aux']:
                root_id_list.append(i)

        return root_id_list

    def _get_del_sample(self, tokens, field, sample):
        del_indexes = []
        for i, token in enumerate(tokens):
            # do not + verb → verb
            if token in ['do', 'does', 'did'] and len(tokens) > i + 2:
                if tokens[i + 1] in ['not', 'n\'t']:
                    root_id_list = self._parse_sentence(tokens)
                    pos_tag = self.processor.get_pos(tokens[i + 2])[0][1]
                    if pos_tag in ['VB', 'VBP', 'VBZ', 'VBG', 'VBD', 'VBN'] or (i + 2) in root_id_list:
                        del_list = [i, i + 1]
                        del_sample = sample
                        for i, index in enumerate(del_list):
                            del_sample = del_sample.delete_field_at_index(field, index - i)
                        return del_sample
            if token in ['not', 'n\'t', 'don', 'didn', 'doesn', 'doesn', 'aren', 'isn', 'wasn', 'weren']:
                return sample.delete_field_at_index(field, i)

        return []

    def _get_add_sample(self, field, tokens, sample):
        root_id_list = self._parse_sentence(tokens)
        if root_id_list:
            check_sentence = self._check_sentence(tokens)
            if check_sentence:
                root_id = root_id_list[0]
                add_sample = self._add_sample(field, tokens, root_id, sample)
                return add_sample
            else:
                return []
        else:
            return []

    def _add_sample(self, field, tokens, root_id, sample):
        if tokens[root_id].lower() in ['is', 'was', 'were', 'am', 'are', '\'s', '\'re', '\'m']:
            add_sample = sample.insert_field_before_index(field, root_id + 1, 'not')
            return add_sample
        if tokens[root_id].lower() in ['being']:
            add_sample = sample.insert_field_before_index(field, root_id, 'not')
            return add_sample

        if tokens[root_id].lower() in ['do', 'does', 'did', 'can', 'could', 'could', 'would', 'will', 'have', 'should']:
            add_sample = sample.insert_field_before_index(field, root_id + 1, 'not')
            return add_sample
        else:
            token_pos = self.processor.get_pos(tokens[root_id])
            trans_sent = []
            if token_pos[0][1] in ['VB', 'VBP', 'VBZ', 'VBG', 'VBD', 'VBN', 'NNS', 'NN']:
                if token_pos[0][1] in ['VB', 'VBP', 'VBG']:
                    neg_word = ['do', 'not']
                if token_pos[0][1] in ['VBD', 'VBN']:
                    neg_word = ['did', 'not']
                else:
                    neg_word = ['does', 'not']
                add_sample = sample
                for i, word in enumerate(neg_word):
                    add_sample = add_sample.insert_field_before_index(field, root_id + i, word)
                return add_sample

            return trans_sent


if __name__ == "__main__":
    sent1 = "The quick brown fox jumps over the lazy dog ."
    data_sample = SASample({'x': sent1, 'y': "negative"})
    swap_ins = SentReverseNeg()
    x = swap_ins.transform(data_sample, n=1)

    for sample in x:
        print(sample.dump())
