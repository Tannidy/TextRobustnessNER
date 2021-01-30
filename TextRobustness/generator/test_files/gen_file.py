import codecs
import os

import numpy as np

from TextRobustness.common.preprocess import TextProcessor
from TextRobustness.component.field import TextField
from TextRobustness.component.sample import SASample
from TextRobustness.transformation import SentenceReplaceTransformation
from TextRobustness.transformation.UT.number import Number
from TextRobustness.transformation.UT.ocr import OcrTransformation


def get_words_num(word_sequences):
    return sum(len(word_seq) for word_seq in word_sequences)


def read_data(fn, verbose=True, column_no=-1):
    word_sequences = list()
    tag_sequences = list()
    with codecs.open(fn, 'r', 'utf-8') as f:
        lines = f.readlines()
    curr_words = list()
    curr_tags = list()
    for k in range(len(lines)):
        line = lines[k].strip()
        if len(line) == 0 or line.startswith('-DOCSTART-'):  # new sentence or new document
            if len(curr_words) > 0:
                word_sequences.append(curr_words)
                tag_sequences.append(curr_tags)
                curr_words = list()
                curr_tags = list()
            continue
        strings = line.split(' ')
        word = strings[0]
        tag = strings[column_no]  # be default, we take the last tag
        curr_words.append(word)
        curr_tags.append(tag)
        if k == len(lines) - 1:
            word_sequences.append(curr_words)
            tag_sequences.append(curr_tags)
    print('Loading from %s: %d samples, %d words.' % (fn, len(word_sequences), get_words_num(word_sequences)))
    return word_sequences, tag_sequences

def deTokenize(sent_list):
    str = ""
    for word in sent_list:
        if str == "":
            str = word
        else:
            str += " " + word
    return str

def transform(word_sequences, tag_seqences):
    new_word_sequences = ['-DOCSTART-']
    new_tag_sequences = ['O']
    for i in range(len(word_sequences)):
        word_seq, tag_seq = word_sequences[i], tag_seqences[i]
        sent1 = deTokenize(word_seq)
        data_sample = SASample({'x': sent1, 'y': "negative"})
        trans = OcrTransformation()
        x = trans.transform(data_sample, n=1)
        for sample in x:
            new_word_sequences.append(TextProcessor().sentence_tokenize(sample.dump()['x']))
            new_tag_sequences.append(tag_seq)
            break
    return new_word_sequences, new_tag_sequences


def datalist_to_file(word_sequences, targets_tag_sequences_test, m, dtset):
    with open(dtset + '_new_test_' + m + '.txt', 'w', encoding='utf-8') as f:
        for i in range(len(word_sequences)):
            if i == 0:
                f.write("-DOCSTART- -X- -X- O\n\n")
                continue
            print(word_sequences[i], targets_tag_sequences_test[i])
            for j in range(len(word_sequences[i])):
                assert len(word_sequences[i]) == len(targets_tag_sequences_test[i]), \
                    "sentence length is " + str(len(word_sequences[i])) + ", while tag length is " +  str(len(targets_tag_sequences_test[i]))
                f.write(word_sequences[i][j] + " O O " + targets_tag_sequences_test[i][j] + "\n")
            f.write("\n")


if __name__ == "__main__":

    data_path = ["test_files/conll2003", "test_files/ACE", "test_files/ontonotes"]
    dataset = ["conll2003", "ACE", "ontonotes"]
    data_list = ['train', 'dev', 'test']
    mode_list = ['test', 'write']
    labels = [['ORG', 'PER', 'LOC', 'MISC'],
              ['ORG', 'LOC', 'PER', 'FAC', 'GPE', 'VEH', 'WEA'],
              ['ORG', 'LOC', 'PERSON', 'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW',
               'MONEY', 'NORP', 'ORDINAL', 'PERCENT', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']]
    transformation = ['number', 'ocr', 'spelling', 'tense', 'typos', 'word_embedding', 'antonym', 'synonym']
    mode = mode_list[0]

    for i in range(0, 3):
        res_path = "res_" + dataset[i] + '/'
        word_sequences, targets_tag_sequences_test = read_data(fn=os.path.join(data_path[i], 'test' + '.txt'))
        new_word_sequences, new_tag_sequences = transform(word_sequences, targets_tag_sequences_test)
        if mode == 'write':
            datalist_to_file(new_word_sequences, new_tag_sequences, transformation[1], dataset[i])
        else:
            new_word_sequences = new_word_sequences[:20]
            new_tag_sequences = new_tag_sequences[:20]
            for i in range(len(new_word_sequences)):
                print(new_word_sequences[i], new_tag_sequences[i])
