"""
Base Sample Abstract Class
============================================

"""
import copy
from abc import ABC, abstractmethod

from TextRobustness.component.field import ListField, TextField


class Sample(ABC):
    """ Base Sample class to hold the necessary info and provide atomic operations.

    Attributes:
        data : json
            THe json obj that contains data info.
        origin: Sample
            Original sample obj.
    """
    def __init__(self, data, origin=None):
        self.origin = origin if origin else self
        self.log = []
        self.check_data(data)
        self.load(data)

    def __repr__(self):
        return 'Sample'

    def free_log(self):
        """Delete log that take up memory.

        """
        self.log = []

    def get_value(self, field):
        """Get field value by field_str.

        Args:
            field: str

        Returns:
            field value
        """
        return copy.deepcopy(getattr(self, field).field_value)

    def get_words(self, field):
        """Get tokenized words of given textfield.

        Args:
            field: str

        Returns:
            list
        """
        field_obj = getattr(self, field)
        assert isinstance(field_obj, TextField)

        return field_obj.words[:]

    def get_mask(self, field):
        field_obj = getattr(self, field)
        assert isinstance(field_obj, (ListField, TextField))

        return field_obj.mask[:]

    def get_sentences(self, field):
        """Get split sentences of given textfield.

        Args:
            field: str

        Returns:
            list
        """
        field_obj = getattr(self, field)
        assert isinstance(field_obj, TextField)

        return field_obj.sentences[:]

    def get_pos(self, field):
        """Get text field pos tag.

        Args:
            field: str

        Returns:
            pos tag list.
        """
        field_obj = getattr(self, field)
        assert isinstance(field_obj, TextField)

        return field_obj.pos_tagging[:]

    def get_ner(self, field):
        """Get text field ner tag.

        Args:
            field: str

        Returns:
            ner tag list.
        """
        field_obj = getattr(self, field)
        assert isinstance(field_obj, TextField)

        return field_obj.ner

    def replace_fields(self, fields, field_values):
        """Fully replace multi fields at the same time and return new sample.

        Args:
            fields: list
                field str list.
            field_values: list
                field value list.

        Returns:
            Sample

        """
        assert len(fields) == len(field_values)
        sample = self.clone(self)

        for index, field in enumerate(fields):
            origin_field = getattr(sample, field)
            assert isinstance(field_values[index], origin_field.field_type)
            setattr(sample, field, origin_field.new_field(field_values[index]))

        return sample

    def replace_field(self, field, field_value):
        """Fully replace single field and return new sample.

        Args:
            field: str
                field str.
            field_value: field_type
                Field value.

        Returns:
            Sample

        """
        return self.replace_fields([field], [field_value])

    def replace_fields_at_indices(self, fields, fields_sub_indices, fields_sub_items):
        """Replace multi indices of ListFields at the same time.

        Stay away from the complex function !!!
        Be careful of your input list shape.

        Args:
            fields: list
                list of fields str
            fields_sub_indices: list of list
                shape：fields_num * indices_num
                different fields corresponding different substitute indices
            fields_sub_items: list of list
                shape: fields_num * indices_num

        Returns:
            Sample

        """
        assert isinstance(fields, list) & isinstance(fields_sub_indices, list) & isinstance(fields_sub_items, list)
        assert len(fields) == len(fields_sub_indices) == len(fields_sub_items)

        for i in range(len(fields)):
            assert len(fields_sub_indices[i]) == len(fields_sub_items[i])

        sample = self.clone(self)

        for index, field in enumerate(fields):
            field_obj = getattr(self, field)
            assert isinstance(field_obj, (ListField, TextField))
            rep_obj = field_obj.replace_at_indices(fields_sub_indices[index], fields_sub_items[index])
            setattr(sample, field, rep_obj)

        return sample

    def replace_field_at_indices(self, field, field_sub_indices, field_sub_items):
        """Replace multi indices of List at the same time.

        Stay away from the complex function !!!
        Be careful of your input list shape.

        Args:
            field: str
                field str
            field_sub_indices: list
                shape：indices_num
            field_sub_items: list of list
                shape: indices_num

        Returns:
            Sample

        """

        return self.replace_fields_at_indices([field], [field_sub_indices], [field_sub_items])

    def delete_field_at_index(self, field, del_index):
        sample = self.clone(self)
        field_obj = getattr(sample, field)

        assert isinstance(field_obj, (ListField, TextField))
        rep_obj = field_obj.delete_at_index(del_index)
        setattr(sample, field, rep_obj)

        return sample

    def insert_field_before_index(self, field,  ins_index, new_item):
        sample = self.clone(self)

        field_obj = getattr(sample, field)
        assert isinstance(field_obj, (ListField, TextField))
        rep_obj = field_obj.insert_before_index(ins_index, new_item)
        setattr(sample, field, rep_obj)

        return sample

    def insert_field_after_index(self, field, ins_index, new_item):
        sample = self.clone(self)

        field_obj = getattr(sample, field)
        assert isinstance(field_obj, (ListField, TextField))
        rep_obj = field_obj.insert_after_index(ins_index, new_item)
        setattr(sample, field, rep_obj)

        return sample

    def swap_field_at_index(self, field, first_index, second_index):
        sample = self.clone(self)

        field_obj = getattr(sample, field)
        assert isinstance(field_obj, (ListField, TextField))
        rep_obj = field_obj.swap_at_index(first_index, second_index)
        setattr(sample, field, rep_obj)

        return sample

    @abstractmethod
    def check_data(self, data):
        """Check rare data format.

        Args:
            data: rare data input.

        """
        raise NotImplementedError

    @abstractmethod
    def load(self, data):
        """Parse data into sample field value.

        Args:
            data: rare data input.

        """
        raise NotImplementedError

    @abstractmethod
    def dump(self):
        """Convert sample info to input data json format.

        Returns:
            Json object.
        """
        raise NotImplementedError

    @staticmethod
    def make_up_log_msg(transformation='', detail=''):
        """Make up log message.

        Args:
            transformation: str
                Transformation name string.
            detail: str
                detail process log.

        Returns:
            log string.
        """
        return "Transformation: {0}, detail: {1}".format(transformation, detail)

    @classmethod
    def clone(cls, original_sample):
        """Deep copy self to a new sample.

        Args:
            original_sample: sample to be copied.
        Returns:
            Sample instance.

        """
        sample = copy.deepcopy(original_sample)
        sample.origin = original_sample.origin

        return sample

    @property
    def is_origin(self):
        """Return whether the sample is original Sample.

        """
        return self.origin is self
