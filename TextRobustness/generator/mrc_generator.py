import nltk
import json
import os
import numpy as np
from TextRobustness.generator import Generator
from sklearn.neighbors import KDTree
from tqdm import tqdm
from collections import Counter
from TextRobustness.component.sample.mrc_sample import MRCSample


class MRCGenerator(Generator):
    def __init__(self, keep_original=False, flatten=True, transformations=None,
                 semantic_validate=False, semantic_score=0.6, max_gen_num=100,
                 word2vec_file=None, save_dir=None, **kwargs):
        super().__init__(task='MRC', keep_original=keep_original, flatten=flatten, transformations=transformations,
                         semantic_validate=semantic_validate, semantic_score=semantic_score, max_gen_num=max_gen_num)
        self.word2vec_file = word2vec_file
        self.save_dir = save_dir

    def _pre_process(self, dataset):
        """Get nearby_word_dict with Glove Embedding"""
        if os.path.exists(self.save_dir+'/neighbour.json'):
            with open(self.save_dir+"/postag_dict.json") as f:
                self.pos_tag_dict = json.load(f)
            with open(self.save_dir+"/neighbour.json") as f:
                self.nearby_word_dict = json.load(f)
        else:
            word_counter = Counter()
            for sample in tqdm(dataset):
                for word in nltk.word_tokenize(sample['x'][0]['context']):
                    word_counter[word] += 1
                for word in nltk.word_tokenize(sample['x'][0]['question']):
                    word_counter[word] += 1
            nearby_words = self.get_neighbour_word(word_counter)
            print("Saving vocabulary at {}".format(self.save_dir))
            with open(self.save_dir + '/neighbour.json', "w") as f:
                json.dump(nearby_words, f)

    def get_neighbour_word(self, words, num_neighbours=20):
        """
        Calculate the distance between words in glove embedding space
        :param words: the vocabulary of dataset
        :param num_neighbours: the num of the nearest words in glove embedding space
        :return: nearby_word_dict
        """
        main_inds = {}
        all_words = []
        all_vecs = []
        with open(self.word2vec_file) as f:
            for i, line in tqdm(enumerate(f)):
                if len(line.rstrip().split(" ")) <= 2:
                    continue
                word, vector = line.rstrip().split(" ", 1)
                vec = np.fromstring(vector, dtype=np.float, sep=" ")
                all_words.append(word)
                all_vecs.append(vec)
                if word in words:
                    main_inds[word] = i
        print("Found vectors for {} / {} words".format(len(main_inds), len(words)))
        tree = KDTree(np.array(all_vecs))
        nearby_words = {}
        for word in tqdm(main_inds):
            dists, inds = tree.query(all_vecs[main_inds[word]].reshape(1, -1),
                                     k=num_neighbours + 1)
            nearby_words[word] = [
                {'word': all_words[i], 'dist': d} for d, i in zip(dists[0], inds[0])]
        return nearby_words

    def generate_sample(self, sample):
        assert isinstance(sample, dict)
        with_aux_x = False if 'aux_x' not in sample.keys() else True

        transformation = self.transform_methods[0]
        trans_instance = transformation(text_processor=self.text_processor)
        transformed_samples = trans_instance.transform(sample, n=1, nearby_word_dict=self.nearby_word_dict,
                                                       pos_tag_dict=self.pos_tag_dict)
        return transformed_samples


if __name__ == "__main__":
    context = 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'
    question = 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?'
    answer_start = 515
    answer_text = 'Saint Bernadette Soubirous'
    sample = {
        'context': context,
        'question': question,
        'answer': answer_text,
        'answer_start': answer_start
    }
    gene = MRCGenerator(save_dir='/home/zxp/textrobustness')
    sample = MRCSample(sample)
    sample = gene.generate(sample)
    print(sample.dump())
