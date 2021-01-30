"""
NLTK tokenize and its reverse tokenize function
============================================

"""

import nltk
import re


# TODO， fix bug of '"' would be convert to '“'
def tokenize(text):
    """ Split a text into tokens (words, morphemes we can separate such as
        "n't", and punctuation).

    Args:
        text: string

    Returns:
        list of tokens

    """
    def _tokenize_gen(text):
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                yield word

    return list(_tokenize_gen(text))


def untokenize(words):
    """ Untokenizing a text undoes the tokenizing operation, restoring
        punctuation and spaces to the places that people expect them to be.
        Ideally, `untokenize(tokenize(text))` should be identical to `text`,
        except for line breaks.

        Watch out!
        Default punctuation add to the word before its index, it may raise inconsistency bug.

    Args:
        words: list

    Returns:
        sentence string.

    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .', '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
        "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()


if __name__ == "__main__":
    sents = ['I dont know this issue.',
             "I don't know this issue,.",
             "2020-year-plan is to write 3 papers",
             'this is ., token ? ! @# . Ok!'
             ]

    for sent in sents:
        print('------------------sent -> word -> sent--------------------------')
        words = tokenize(sent)
        print(words)
        print(sent)
        print(untokenize(words))

        print('------------------word -> sent -> word--------------------------')
        words = sent.split(' ')
        print(words)
        print(sent)
        print(untokenize(words))
        print('*****************************************************************')

    print(untokenize(['This is a paragraph.',  "new sentence, ", 'wow']))
