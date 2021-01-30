"""
UT Generator Class
============================================

"""
import codecs
import os

from TextRobustness.common.preprocess import TextProcessor
from TextRobustness.generator import Generator, UTGenerator

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

def datalist_to_file(word_sequences, targets_tag_sequences_test, m, dtset):
    with open(dtset + '_new_test_' + m + '.txt', 'w', encoding='utf-8') as f:
        for i in range(len(word_sequences)):
            if i == 0:
                f.write("-DOCSTART- -X- -X- O\n\n")
                continue
            print(word_sequences[i], targets_tag_sequences_test[i])
            for j in range(len(word_sequences[i])):
                assert len(word_sequences[i]) == len(targets_tag_sequences_test[i]), \
                    "sentence length is " + str(len(word_sequences[i])) + \
                    ", while tag length is " +  str(len(targets_tag_sequences_test[i])) + \
                    str(word_sequences[i]) + str(targets_tag_sequences_test[i])
                f.write(word_sequences[i][j] + " O O " + targets_tag_sequences_test[i][j] + "\n")
            f.write("\n")

if __name__ == "__main__":
    from TextRobustness.dataset import APICallDataset

    data_path = ["test_files/conll2003", "test_files/ACE", "test_files/ontonotes"]
    dataset_choice = ["conll2003", "ACE", "ontonotes"]
    data_list = ['train', 'dev', 'test']
    # mode_list = ['test', 'write']
    labels = [['ORG', 'PER', 'LOC', 'MISC'],
              ['ORG', 'LOC', 'PER', 'FAC', 'GPE', 'VEH', 'WEA'],
              ['ORG', 'LOC', 'PERSON', 'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW',
               'MONEY', 'NORP', 'ORDINAL', 'PERCENT', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']]
    # # transformation = ['number', 'ocr', 'spelling', 'tense', 'typos', 'word_embedding', 'antonym', 'synonym']
    # mode = mode_list[0]

    for i in range(0, 3):
        res_path = "res_" + dataset_choice[i] + '/'
        word_sequences, tag_sequences = read_data(fn=os.path.join(data_path[i], 'test' + '.txt'))
        sent_list = [deTokenize(sent) for sent in word_sequences]
        # sent1 = 'The fast brown fox jumps over the lazy dog .'
        # sent2 = 'We read the world wrong and say it deceives us.'
        # load data and create task samples automatically
        dataset = APICallDataset({'x': sent_list}, task='UT')
        gene = UTGenerator()

        for trans_rst, trans_type in gene.generate(dataset):
            print("------------This is {0} transformation!----------".format(trans_type))
            new_word_sequences = ['<>']

            for sample in trans_rst:
                sent = sample.dump()['x']
                sent_to_list = TextProcessor().sentence_tokenize(sent)
                new_list = []
                for token in sent_to_list:
                    new_list+=token.split(' ')
                new_word_sequences.append(new_list)
                # print(sample.dump())
            new_word_sequences = new_word_sequences[1:]
            assert len(new_word_sequences) == len(tag_sequences)
            datalist_to_file(new_word_sequences, tag_sequences, trans_type, dataset_choice[i])


