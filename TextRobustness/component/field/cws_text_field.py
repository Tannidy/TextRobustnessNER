from TextRobustness.component.field.text_field import TextField
from TextRobustness.common.preprocess.cws_text_processor import cws_text_processor


class CWSTextField(TextField):
    """
        A helper class that represents input string that to be modified.

        Text that Sample contains parsed in data set,
        ``TextField`` provides multiple methods for Sample to modify

        Attributes:
            field_value: list
                Tokenized words list.
        """
    def __init__(self, field_value):
        if isinstance(field_value, str):
            sentence = field_value
        elif isinstance(field_value, list):
            # join and re-tokenize because of insert/delete operation
            sentence = ''
            for word in field_value:
                sentence += word
        else:
            raise ValueError('TextField supports string/token list, given {0}'.format(type(field_value)))
        super().__init__(sentence)
        self._ner_tags = None
        self._ner_list = None
        self._pos_tag = None
        self.token = sentence

    def ner(self):
        if not self._ner_tags:
            ner_tags, self._ner_list = cws_text_processor.get_ner(self.field_value)
            if len(self._ner_list) != len(self.field_value):
                raise ValueError(f"POS tagging not aligned with tokenized words")
            self._ner_tags = ner_tags
        return self._ner_tags, self._ner_list

    def pos_tag(self):
        if not self._pos_tag:
            self._pos_tag = cws_text_processor.get_pos_tag(self.field_value)
        return self._pos_tag
