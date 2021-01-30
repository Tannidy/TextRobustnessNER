import copy
import numpy as np


def trade_off_sub_words(sub_words, sub_indices, trans_num):
    """ Select proper candidate words to maximum number of transform result.

    Select words of top n substitutes words number.
    TODO, add random mechanism

    Args:
        sub_words: list of substitutes word of each legal word
        sub_indices: list of indices of each legal word
        trans_num: max number of words to apply substitution

    Returns:
        sub_words after alignment + indices of sub_words

    """
    # max number of words to apply transform
    trans_num = min(len(sub_words), trans_num)

    sub_num = [len(words) for words in sub_words]
    re_sub_indices = np.array(sub_num).argsort()[::-1].tolist()[:trans_num]

    final_sub_words = []
    final_sub_indices = []

    # filter 0 candidates indices
    for index in re_sub_indices:
        if len(sub_words[index]) > 0:
            final_sub_words.append(sub_words[index])
            final_sub_indices.append(sub_indices[index])
        else:
            break

    # align sub words number because different word with different candidates
    if final_sub_words:
        min_sub = len(final_sub_words[-1])
        final_sub_words = [final_sub_word[:min_sub] for final_sub_word in final_sub_words]

    return final_sub_words, final_sub_indices


def replace_items_at_ranges(origin_list, ranges, new_items):
    """Replace items by ranges.

    Args:
        origin_list: list
            the list value to be modified.
        ranges: int/list/slice
            can be int indicate replace single item or their list like [1, 2, 3].
            can be list like (0,3) indicate replace items from 0 to 3(not included)
                or their list like [(0, 3), (5,6)]
            can be slice which would be convert to list or their list.
            Watch out! Each range must be the same type!
        new_items: list
            items corresponding ranges.

    Returns:
        new list

    """
    items = copy.deepcopy(origin_list)
    ranges = copy.deepcopy(ranges)

    if len(ranges) == 1 and isinstance(ranges[0], slice):
        new_items = new_items[0]
        ranges = list(range(len(items))[ranges[0]])

    if new_items is None:
        new_items = [None] * len(ranges)

    if len(new_items) != len(ranges):
        raise ValueError(
            f"Cannot replace {len(new_items)} tokens at {len(ranges)} ranges."
        )

    # ranges check
    for idx, range_i in enumerate(ranges):
        if not isinstance(range_i, list) and not isinstance(range_i, tuple) and not isinstance(range_i, int):
            raise TypeError(
                f"replace_items_at_ranges requires list of ``list``, ``tuple`` or ``int``, got {type(range_i)}"
            )
        if isinstance(range_i, int):
            if (range_i < 0) or (range_i > len(items) - 1):
                raise ValueError(
                    f"Can't replace {range_i} index of {len(items)} length"
                )
            ranges[idx] = range_i = (range_i, range_i + 1)

        if not all(isinstance(bound, int) for bound in range_i):
            raise TypeError(
                f"replace_items_at_ranges requires ``int`` elements, got {range_i} at index {idx}"
            )
        if len(range_i) > 2:
            raise ValueError(
                f"replace_items_at_ranges requires range not longer than 2, got {len(range_i)} at index {idx}"
            )
        range_i = list(range_i)
        range_i[0] = max(0, range_i[0])
        range_i[1] = min(len(items), range_i[1])
        ranges[idx] = range_i

        if range_i[0] >= range_i[1]:
            raise ValueError(
                f"No elements selected in range between {range_i[0]} and {range_i[1]}"
            )

    # check whether exist range collision
    def check_collision(r):
        for i, range1 in enumerate(r):
            for j, range2 in enumerate(r[i + 1:]):
                l1, r1 = range1
                l2, r2 = range2
                if max(l1, l2) < min(r1, r2):
                    return True
        return False

    if check_collision(ranges):
        raise ValueError(
            f"Ranges has collision"
        )
    real_items = []

    for idx, item in enumerate(new_items):
        next_item = item
        if not isinstance(item, list):
            next_item = [item]
        if item in [None, '', []]:  # Assume token is empty if it's ``None``, ``[]``, ``''``
            next_item = []
        real_items.append(next_item)

    sorted_items, sorted_ranges = zip(*sorted(zip(real_items, ranges), key=lambda x: x[1]))
    replaced_items = items[:sorted_ranges[0][0]]
    sorted_ranges = list(sorted_ranges)
    sorted_ranges.append([len(items), -1])

    for idx, sorted_token in enumerate(sorted_items):
        replaced_items.extend(sorted_token)
        replaced_items.extend(items[sorted_ranges[idx][1]: sorted_ranges[idx + 1][0]])

    return replaced_items


def replace_at_ranges(origin_list, ranges, new_items):
    """ Replace items of given list instance.

    Args:
        origin_list: list
        ranges: int/list/slice
            can be int indicate replace single item or their list like [1, 2, 3].
            can be list like (0,3) indicate replace items from 0 to 3(not included)
                or their list like [(0, 3), (5,6)]
            can be slice which would be convert to list or their list.
            Watch out! Each range must be the same type!
        new_items: list
            items corresponding ranges.

    Returns:
        new list

    """
    
    return replace_items_at_ranges(origin_list, ranges, new_items)


def replace_at_indices(origin_list, indices, new_items):
    return replace_at_ranges(origin_list, indices, new_items)


def replace_at_index(origin_list, index, new_token):
    return replace_at_ranges(origin_list, [index], [new_token])


def delete_at_index(origin_list, index):
    return replace_at_ranges(origin_list, [index], [None])


def insert_before_index(origin_list, index, new_item):
    new_items = new_item
    if not isinstance(new_item, list):
        new_items = [new_item]
    new_items.extend([origin_list[index]])

    return replace_at_ranges(origin_list, [index], [new_items])


def insert_after_index(origin_list, index, new_item):
    new_items = [origin_list[index]]
    if not isinstance(new_item, list):
        new_item = [new_item]
    new_items.extend(new_item)

    return replace_at_ranges(origin_list, [index], [new_items])


def swap_at_index(origin_list, first_index, second_index):
    if max(first_index, second_index) > len(origin_list) - 1:
        raise ValueError(f"Can't swap {0} and {1} index to items {2} of {3} length"
                         .format(first_index, second_index, origin_list, len(origin_list)))

    return replace_at_ranges(origin_list, [first_index, second_index],
                             [origin_list[second_index], origin_list[first_index]])
