import random
import string


def get_start_end(word, skip_first=False, skip_last=False):
    """ Get valid operation range of one word.

    Args:
        word: target word string
        skip_first: whether operate first char
        skip_last: whether operate last char

    Returns:
        start index, last index
    """
    chars = list(word)
    start = int(skip_first)
    end = len(chars) - 1 - int(skip_last)

    return start, end


def get_random_letter(src_char=None):
    """ Get replaced letter according src_char format.

    Args:
        src_char: char

    Returns:
        char
            default return a lower letter.

    """
    if src_char.isdigit():
        return random.choice(string.digits)
    if src_char.isupper():
        return random.choice(string.ascii_uppercase)
    else:
        return random.choice(string.ascii_lowercase)


def swap(word,  num=1, skip_first=False, skip_last=False):
    """ Swaps random characters with their neighbors.

    Args:
        word: str
            target word
        num: int
            number of typos to add
        skip_first: bool
            whether swap first char of word
        skip_last: bool
             whether swap last char of word

    Returns:
        list(string)
            perturbed strings
    """
    if len(word) <= 1:
        return word

    chars = list(word)
    start, end = get_start_end(word, skip_first, skip_last)

    # error swap num, return original word
    if end - start < num:
        return None

    swap_idxes = random.sample(list(range(start, end)), num)

    for swap in swap_idxes:
        tmp = chars[swap]
        chars[swap] = chars[swap + 1]
        chars[swap + 1] = tmp

    return ''.join(chars)


def insert(word, num=1, skip_first=False, skip_last=False):
    """ Perturb the word with 1 random character inserted.

    Args:
        word: str
            target word
        num: int
            number of typos to add
        skip_first: bool
            whether insert char at the beginning of word
        skip_last: bool
            whether insert char at the end of word

    Returns:
        list(string)
            perturbed strings

    """
    if len(word) <= 1:
        return word

    chars = list(word)
    start, end = get_start_end(word, skip_first, skip_last)

    if end - start + 2 < num:
        return None

    swap_idxes = random.sample(list(range(start, end+2)), num)
    swap_idxes.sort(reverse=True)

    for idx in swap_idxes:
        insert_char = get_random_letter(chars[min(idx, len(chars)-1)])
        chars = chars[:idx] + [insert_char] + chars[idx:]

    return "".join(chars)


def delete(word, num=1, skip_first=False, skip_last=False):
    """ Perturb the word with 1 letter deleted.

    Args:
        word: str
            target word
        num: int
            number of typos to add
        skip_first: bool
            whether delete the char at the beginning of word
        skip_last: bool
            whether delete the char at the end of word

    Returns:
        list(string)
            perturbed strings

    """
    if len(word) <= 1:
        return word

    chars = list(word)
    start, end = get_start_end(word, skip_first, skip_last)

    if end - start + 1 < num:
        return None

    swap_idxes = random.sample(list(range(start, end + 1)), num)
    swap_idxes.sort(reverse=True)

    for idx in swap_idxes:
        chars = chars[:idx] + chars[idx + 1:]

    return "".join(chars)


def replace(word, num=1, skip_first=False, skip_last=False):
    """ Perturb the word with 1 letter substituted for a random letter.

    Args:
        word: str
            target word
        num: int
            number of typos to add
        skip_first: bool
            whether replace the char at the beginning of word
        skip_last: bool
            whether replace the char at the end of word

    Returns:
        list(string)
            perturbed strings

    """
    if len(word) <= 1:
        return []

    chars = list(word)
    start, end = get_start_end(word, skip_first, skip_last)

    # error swap num, return original word
    if end - start + 1 < num:
        return word

    idxes = random.sample(list(range(start, end+1)), num)

    for idx in idxes:
        chars[idx] = get_random_letter(chars[idx])

    return "".join(chars)
