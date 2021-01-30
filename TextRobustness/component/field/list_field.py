"""
List Field Class
=====================

A helper class that represents input list values that to be modified.
"""

from TextRobustness.component.field import Field
from TextRobustness.common.utils.list_op import *
from TextRobustness.common.settings import ORIGIN, TASK_MASK, MODIFIED_MASK


class ListField(Field):
    """A helper class that represents input list values that to be modified.

    Operations which modify field_value would generate new Field instance.

    Args:
        field_value: list of str
            The list that ListField represents.
    """

    def __init__(self, field_value, mask=None):
        if isinstance(field_value, str):
            field_value = list(field_value)
        super().__init__(field_value, field_type=list)

        if not mask:
            self._mask = [ORIGIN] * len(field_value)
        else:
            assert len(mask) == len(self.field_value)
            for mask_item in mask:
                if mask_item not in [ORIGIN, TASK_MASK, MODIFIED_MASK]:
                    raise ValueError("Not support mask value of {0}".format(mask_item))
            self._mask = mask

    @property
    def mask(self):
        return self._mask[:]

    def set_mask(self, index, value):
        if index > len(self._mask) - 1 or index < 0:
            raise ValueError("Index {0} out of range {1}".format(index, len(self._mask) - 1))
        if value not in [ORIGIN, TASK_MASK, MODIFIED_MASK]:
            raise ValueError('Support mask value in {0}, '
                             'while input mask value is {1}!'.format([ORIGIN, TASK_MASK, MODIFIED_MASK], value))
        self._mask[index] = value

    def replace_at_indices(self, indices, new_items):
        """ Replace items at indices and set their mask to MODIFIED_MASK. """
        new_mask = replace_at_indices(self.mask, indices, [MODIFIED_MASK] * len(indices))
        new_field = replace_at_indices(self.field_value, indices, new_items)

        return self.new_field(new_field, mask=new_mask)

    def replace_at_index(self, index, new_token):
        """ Replace item at index and set its mask to MODIFIED_MASK. """
        return self.replace_at_indices([index], [new_token])

    def delete_at_index(self, index):
        """ Delete item at index and remove their mask value. """
        new_mask = delete_at_index(self.mask, index)
        new_field = delete_at_index(self.field_value, index)

        return self.new_field(new_field, mask=new_mask)

    def insert_before_index(self, index, new_item):
        """ Insert item before index and add MODIFIED_MASK to mask list. """
        mask_value = [MODIFIED_MASK]*len(new_item) if isinstance(new_item, list) else MODIFIED_MASK
        new_mask = insert_before_index(self.mask, index, mask_value)
        new_field = insert_before_index(self.field_value, index, new_item)

        return self.new_field(new_field, mask=new_mask)

    def insert_after_index(self, index, new_item):
        """ Insert item after index and add MODIFIED_MASK to mask list. """
        mask_value = [MODIFIED_MASK] * len(new_item) if isinstance(new_item, list) else MODIFIED_MASK
        new_mask = insert_after_index(self.mask, index, mask_value)
        new_field = insert_after_index(self.field_value, index, new_item)

        return self.new_field(new_field, mask=new_mask)

    def swap_at_index(self, first_index, second_index):
        """ Swap item between first_index and second_index and modify their mask value. """
        new_mask = replace_at_indices(self.mask, [first_index, second_index], [MODIFIED_MASK] * 2)
        new_field = swap_at_index(self.field_value, first_index, second_index)

        return self.new_field(new_field, mask=new_mask)

    def __len__(self):
        return len(self.field_value)

    def __getitem__(self, key):
        return self.field_value[key]


if __name__ == "__main__":
    x = ListField(['H', 'e', 'l', 'l', 'o', ',', ' ', 'h', 'a', 'p', 'p', 'y', ' ', 'w', 'o', 'l', 'r', 'd', '!'])
    print('_'.join(x.field_value))
    print('--------------------test operations-------------------------')

    # test insert
    print('_'.join(x.field_value))
    insert_before = x.insert_before_index(0, ['H', 'e', 'l', 'l', 'o', ',', ' '])
    print('_'.join(insert_before.field_value))
    insert_after = x.insert_after_index(len(x) - 1, ['B', 'y', 'e', '!'])
    print('_'.join(insert_after.field_value))

    # test swap
    swap = x.swap_at_index(0, 1)
    print('_'.join(x.field_value))
    print('_'.join(swap.field_value))

    # test delete
    delete = x.delete_at_index(0)
    print('_'.join(x.field_value))
    print('_'.join(delete.field_value))

    # test replace
    replace = x.replace_at_index(0, '$')
    print('_'.join(x.field_value))

