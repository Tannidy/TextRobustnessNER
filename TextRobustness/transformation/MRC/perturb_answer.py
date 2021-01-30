"""
Perturb Answer by altering the sentence that contains answer
==========================================================
"""
import json
import collections

from TextRobustness.transformation import Transformation
from TextRobustness.component import Field
from TextRobustness.component.field import TextField
from TextRobustness.component.sample.mrc_sample import MRCSample
from TextRobustness.common.utils.feature_extract import *


class PerturbAnswer(Transformation):
    """Transform the sentence containing answer with AlterSentence transformation."""

    def __init__(self, **kwargs):
        super().__init__()

        # Rules for altering sentences
        self.feat = CorenlpFeature()
        self.rules = collections.OrderedDict([
            # ('special', alter_special),                                            # special tokens transformation
            ('wn_synonyms', MRCSample.alter_wordnet_synonyms),                               # synonym words in WordNet
            ('nearbyProperNoun', MRCSample.alter_nearby(['NNP', 'NNPS'])),                   # proper nouns
            ('nearbyProperNoun', MRCSample.alter_nearby(['NNP', 'NNPS'], ignore_pos=True)),
            ('nearbyEntityNouns', MRCSample.alter_nearby(['NN', 'NNS'], is_ner=True)),       # entity nouns
            ('nearbyEntityJJ', MRCSample.alter_nearby(['JJ', 'JJR', 'JJS'], is_ner=True)),   # entity type
        ])

    def _transform(self, sample, field=None, n=5, nearby_word_dict=None, pos_tag_dict=None, **kwargs):
        """Extract the sentence with answer from context, replace synonyms based on WordNet and glove
        embedding space while keep the semantic meaning unchanged.

        Args:
            sample: the dataset to transform
            nearby_word_dict: the dictionary to search for nearby words
            pos_tag_dict: the dictionary to search for the most frequent pos tags
            **kwargs:

        Returns:
            dict, data structure like follows
            {'x': [{"context": context, "question": question}],
            'y': [answer]
            }

        """
        answer_start = sample.answer_start.field_value
        answer_text = sample.answer.field_value
        sentences = sample.context.sentences
        # filter no-answer samples
        if answer_start < 0:
            return []

        length = 0
        alter_id = None
        # Pick up the sentence that contains the answer
        for i, sent in enumerate(sentences):
            if length + len(sent) < answer_start:
                length = length + len(sent) + 1
                continue
            if sent.find(answer_text) < 0:
                return []
            # Replace the answer with a mask
            sent = sent.replace(answer_text, "ansunk")
            try:
                sent, _ = self.feat.feature_extract(sent)
            except:
                return []
            # Transform a sentence with AlterSentence function
            alter_sent, _ = sample.alter_sentence(sent,
                                                  nearby_word_dict=nearby_word_dict,
                                                  pos_tag_dict=pos_tag_dict,
                                                  rules=self.rules
                                                  )
            # keep the answer unchanged
            alter_sent = alter_sent.replace("ansunk", answer_text)
            new_answer_start = alter_sent.find(answer_text) + length
            alter_id = i
            break
        transform_samples = []
        sentences[alter_id] = alter_sent
        new_context = " ".join(sentences)
        try:
            assert new_context[new_answer_start:new_answer_start + len(answer_text)] == answer_text
        except AssertionError:
            return []
        new_context_field = TextField(new_context)
        new_start_field = Field(new_answer_start)
        new_sample = sample.replace_fields(['context', 'answer_start'],
                                           [new_context_field, new_start_field])
        new_sample.context._sentences = sentences
        transform_samples.append(new_sample)

        return transform_samples


if __name__ == '__main__':
    save_dir = '/home/zxp/textrobustness'
    with open(save_dir + "/postag_dict.json") as f:
        pos_tag_dict = json.load(f)
    with open(save_dir + "/neighbour.json") as f:
        nearby_word_dict = json.load(f)
    context = 'Architecturally, the school has a Catholic character. ' \
              'Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. ' \
              'Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". ' \
              'Next to the Main Building is the Basilica of the Sacred Heart. ' \
              'Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. ' \
              'It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'
    question = 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?'
    answer_start = 515
    answer_text = 'Saint Bernadette Soubirous'
    sample = {
        'context': context,
        'question': question,
        'answer': answer_text,
        'answer_start': answer_start
    }
    sample = MRCSample(sample)
    mp = PerturbAnswer()
    samples = mp.transform(sample, nearby_word_dict=nearby_word_dict,
                           pos_tag_dict=pos_tag_dict)
    for sample in samples:
        print(sample.dump())
