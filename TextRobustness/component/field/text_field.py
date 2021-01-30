"""
Text Field Class
=====================

A helper class that represents input string that to be modified.
"""

from TextRobustness.component.field import Field
from TextRobustness.common.utils.list_op import *
from TextRobustness.common.preprocess.en_processor import text_processor
from TextRobustness.common.settings import ORIGIN, TASK_MASK, MODIFIED_MASK


class TextField(Field):
    """A helper class that represents input string that to be modified.

    Text that Sample contains parsed in data set,
    ``TextField`` provides multiple methods for Sample to modify.

    Support sentence level and word level modification, default using word level API.

    Attributes:
        field_value: str
            Sentence string.

    """

    def __init__(self, field_value, mask=None):
        if isinstance(field_value, str):
            self._words = text_processor.word_tokenize(field_value)
        elif isinstance(field_value, list):
            self._words = field_value
            field_value = text_processor.inverse_tokenize(field_value)
        else:
            raise ValueError('TextField supports string/token list, given {0}'.format(type(field_value)))

        super().__init__(field_value, field_type=str)

        if not mask:
            self._mask = [ORIGIN] * len(self._words)
        else:
            assert len(mask) == len(self.words)
            for mask_item in mask:
                if mask_item not in [ORIGIN, TASK_MASK, MODIFIED_MASK]:
                    raise ValueError("Not support mask value of {0}".format(mask_item))
            self._mask = mask

        # Process tags lazily
        self._sentences = None
        self._pos_tags = None
        self._ner_tags = None
        self._dp_tags = None

    def __hash__(self):
        return hash(self.text)

    @property
    def mask(self):
        return self._mask[:]

    def set_mask(self, index, value):
        if index > len(self._mask) - 1:
            raise ValueError("Index {0} out of range {1}".format(index, len(self._mask) - 1))
        if value not in [ORIGIN, TASK_MASK, MODIFIED_MASK]:
            raise ValueError('Support mask value in {0}, '
                             'while input mask value is {1}!'.format([ORIGIN, TASK_MASK, MODIFIED_MASK], value))
        self._mask[index] = value

    def pos_of_word_index(self, desired_word_idx):
        if (desired_word_idx < 0) or (desired_word_idx > len(self.field_value)):
            raise ValueError(f"Cannot get POS tagging at index {desired_word_idx}")

        return self.pos_tagging[desired_word_idx]

    def replace_at_indices(self, indices, new_items):
        """ Replace words at indices and set their mask to MODIFIED_MASK.

        Args:
            indices: list
            new_items: list

        Returns:
            Replaced TextField object.
        """
        new_mask = replace_at_indices(self.mask, indices, [MODIFIED_MASK]*len(indices))
        new_field = replace_at_indices(self._words, indices, new_items)

        return self.new_field(new_field, mask=new_mask)

    def replace_at_index(self, index, new_token):
        """ Replace words at indices and set their mask to MODIFIED_MASK.

        Args:
            index: int
            new_token: str

        Returns:
            Replaced TextField object.
        """
        return self.replace_at_indices([index], [new_token])

    def delete_at_index(self, index):
        """ Delete word at index and remove their mask value.

        Args:
            index: int

        Returns:
            Modified TextField object.
        """
        new_mask = delete_at_index(self.mask, index)
        new_field = delete_at_index(self._words, index)

        return self.new_field(new_field, mask=new_mask)

    def insert_before_index(self, index, new_item):
        """ Insert word before index and remove their mask value.

        Args:
            index: int
            new_item: str

        Returns:
            Modified TextField object.
        """
        mask_value = [MODIFIED_MASK] * len(new_item) if isinstance(new_item, list) else MODIFIED_MASK
        new_mask = insert_before_index(self.mask, index, mask_value)
        new_field = insert_before_index(self._words, index, new_item)

        return self.new_field(new_field, mask=new_mask)

    def insert_after_index(self, index, new_item):
        mask_value = [MODIFIED_MASK] * len(new_item) if isinstance(new_item, list) else MODIFIED_MASK
        new_mask = insert_before_index(self.mask, index, mask_value)
        new_field = insert_after_index(self._words, index, new_item)

        return self.new_field(new_field, mask=new_mask)

    def swap_at_index(self, first_index, second_index):
        new_mask = replace_at_indices(self.mask, [first_index, second_index], [MODIFIED_MASK]*2)
        new_field = swap_at_index(self._words, first_index, second_index)

        return self.new_field(new_field, mask=new_mask)

    @staticmethod
    def get_word_case(word):
        if len(word) == 0:
            return 'empty'

        if len(word) == 1 and word.isupper():
            return 'capitalize'

        if word.isupper():
            return 'upper'
        elif word.islower():
            return 'lower'
        else:
            for i, c in enumerate(word):
                if i == 0:  # do not check first character
                    continue
                if c.isupper():
                    return 'mixed'

            if word[0].isupper():
                return 'capitalize'
            return 'unknown'

    @property
    def words(self):
        return self._words

    @property
    def sentences(self):
        if not self._sentences:
            self._sentences = text_processor.sentence_tokenize(self.field_value)

        return self._sentences

    @property
    def text(self):
        return self.field_value

    @property
    def pos_tagging(self):
        if not self._pos_tags:
            pos_tags = [pos for w, pos in text_processor.get_pos(self.words)]
            if len(pos_tags) != len(self._words):
                raise ValueError(f"POS tagging not aligned with tokenized words")
            self._pos_tags = pos_tags

        return self._pos_tags

    @property
    def ner(self):
        """ Get NER tags.

        Example:
            input sentence 'Lionel Messi is a football player from Argentina.'

            >>[('Lionel Messi', 0, 2, 'PERSON'),
               ('Argentina', 7, 8, 'LOCATION')]

        Returns:
            A list of tuples, *(entity, start, end, label)*

        """
        if not self._ner_tags:
            self._ner_tags = text_processor.get_ner(self.words, return_char_idx=False)

        return self._ner_tags

    @property
    def dependency_parsing(self):
        if not self._dp_tags:
            self._dp_tags = text_processor.get_dep_parser(self.field_value)

        return self._dp_tags


if __name__ == "__main__":
    x = TextField("Fudan University natural language processing group. Shanghai yangpu area.")
    print(x.field_value, x.words, x.sentences)

    print(x.pos_tagging)
    # test ner
    print(x.ner)
    print(x.dependency_parsing)
    print('')

    print('--------------------test operations-------------------------')
    # test insert
    print(x.field_value)
    insert_before = x.insert_before_index(0, ['test '])
    print(insert_before.field_value)
    insert_after = x.insert_after_index(len(x.words) - 1, [' haha', 'test'])
    print(insert_after.field_value)

    # test swap
    swap = x.swap_at_index(0, 1)
    print(x.field_value)
    print(swap.field_value)

    # test delete
    delete = x.delete_at_index(0)
    print(x.field_value)
    print(delete.field_value)

    # test replace
    replace = x.replace_at_index(0, '$')
    print(x.field_value)
    print(replace.field_value)


