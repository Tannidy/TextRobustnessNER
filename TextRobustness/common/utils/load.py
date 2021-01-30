import urllib
import zipfile
import pkgutil
import pickle
import json
import os
import io
import pandas as pd
import json


def make_zip_downloader(URL, file_list=None):
    """
    This function is used to make a zipfile downloader for data.
    """
    if isinstance(file_list, str):
        file_list = [file_list]

    def DOWNLOAD(path):
        with urllib.request.urlopen(URL) as f:
            zf = zipfile.ZipFile(io.BytesIO(f.read()))
            os.makedirs(path, exist_ok=True)
            if file_list is not None:
                for file in file_list:
                    zf.extract(file, path)
            else:
                zf.extractall(path)
        return

    return DOWNLOAD


def pickle_loader(path):
    return pickle.load(open(path, "rb"))


def url_downloader(url):
    def DOWNLOAD(path):
        with urllib.request.urlopen(url) as f:
            open(path, "wb").write(f.read())
        return True

    return DOWNLOAD


def module_loader(dir_path, filter_str=''):
    assert os.path.exists(dir_path)

    for module in pkgutil.iter_modules([dir_path]):
        # filter illegal module
        if filter_str != '' and module.name.find(filter_str) == -1:
            continue

        yield module.module_finder.find_loader(module.name)[0].load_module()


def task_class_load(pkg_path, task_list, base_class, filter_str=''):
    modules = module_loader(pkg_path, filter_str)
    task_class_map = {}

    for module in modules:
        task = module.__name__.split('_')[0]
        task_class = None
        assert task in task_list

        for attr in dir(module):
            reference = getattr(module, attr)
            if type(reference).__name__ not in ['classobj', 'ABCMeta']:
                continue
            if issubclass(reference, base_class) and reference != base_class:
                task_class = reference
                break

        if task_class is None:
            raise ImportError('Not find task config in {0}, '
                              'plz insure your implementation class extend base Class.'.format(module.name))
        task_class_map[task] = task_class

    return task_class_map


def pkg_class_load(pkg_path, base_class, filter_str=''):
    modules = module_loader(pkg_path, filter_str)
    subclasses = {}

    for module in modules:
        for attr in dir(module):
            reference = getattr(module, attr)
            if type(reference).__name__ not in ['classobj', 'ABCMeta']:
                continue
            if issubclass(reference, base_class) and reference != base_class:
                subclasses[attr] = reference

    return subclasses


def read_data(path):
    """
        read data
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            yield line[:-1]


def cws_get_list(path):
    data = read_data(path)
    res = []
    for t in data:
        res.append(t)
    return res


def sa_dict_loader(path):
    name_dict = {}
    info_csv = pd.read_csv(path,
                           names=['name', 'summary'])
    for row in info_csv.iterrows():
        name_dict[row[1]['name']] = str(row[1]['summary'])
    max_len = -1
    for item in name_dict.keys():
        item_len = len(item.split())
        if item_len > max_len:
            max_len = item_len
    return name_dict, max_len


# -------------------------ABSA load-------------------------
def absa_dict_loader(path, dataset):
    infile = os.path.join(path, './{}/train_sent_towe.json'.format(dataset))
    with open(infile, 'r', encoding='utf-8') as fw:
        examples = json.load(fw)
    return examples



# -------------------------NER load-------------------------
def load_oov_entities(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        dic = {'PER': [], 'ORG': [], 'LOC': [], 'MISC': []}
        for line in lines:
            line = line.strip().split(' ')
            entity = line[0]
            for i in range(1, len(line) - 1):
                entity += ' ' + line[i]
            dic[line[len(line) - 1]] += [entity]
    return dic


def read_cross_entities(path):
    dic = {}

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            label = line[-1:][0]
            word = line[0]
            line = line[1:-1]
            for i in line:
                word += ' ' + i
            if label not in dic:
                dic[label] = [word]
            else:
                dic[label].append(word)

    return dic


# -------------------------UT load---------------------------
# UT Word_Embedding
def load_embedding_words(path):
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f:
            sim_dic = json.loads(line)
        return sim_dic


# UT WordSwapTense
def load_verb_words(path):
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f:
            verb_dic = json.loads(line)
        return verb_dic


def auto_create_path(file_path):
    """ Check file path and create dir path automatically.

    Args:
        file_path:

    Returns:

    """
    if not os.path.exists(file_path):
        dir_path = os.path.dirname(file_path)

        if not os.path.exists(dir_path):
            print('Dir not exists, create automatically.')
            os.makedirs(file_path)


# UT Word_Add_Adverb
def load_adverb_words(path):
    with open(path) as word_file:
        adverb_word_list = list(word_file.read().split())

    return adverb_word_list


# UT WordSwapTwitter
def load_twitter_words(path):
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f:
            verb_dic = json.loads(line)
    return verb_dic


# UT SentAddSent
def load_sentences(path):
    with open(path, encoding='utf-8') as word_file:
        adverb_word_list = list(word_file.read().split())

    return adverb_word_list
