"""
Add a distractor sentence to penalize MRC model
==========================================================
"""
import collections

from TextRobustness.transformation import Transformation
from TextRobustness.common.utils.feature_extract import *
from TextRobustness.component import Field
from TextRobustness.component.field import TextField
from TextRobustness.component.sample.mrc_sample import MRCSample, ConstituencyParse


class AddSentenceDiverse(Transformation):
    """Generate a distractor with altered question and fake answer."""

    def __init__(self, **kwargs):
        super().__init__()
        self.feat = CorenlpFeature()
        self.rules = collections.OrderedDict([
            ('special', MRCSample.alter_special),                                       # special tokens transformation
            ('wn_antonyms', MRCSample.alter_wordnet_antonyms),                               # synonym words in wordnet
            ('nearbyNum', MRCSample.alter_nearby(['CD'], ignore_pos=True)),                  # num
            ('nearbyProperNoun', MRCSample.alter_nearby(['NNP', 'NNPS'])),                   # proper nouns
            ('nearbyProperNoun', MRCSample.alter_nearby(['NNP', 'NNPS'], ignore_pos=True)),
            ('nearbyEntityNouns', MRCSample.alter_nearby(['NN', 'NNS'], is_ner=True)),       # entity nouns
            ('nearbyEntityJJ', MRCSample.alter_nearby(['JJ', 'JJR', 'JJS'], is_ner=True)),   # entity type
            ('entityType', MRCSample.alter_entity_type),
        ])

    def _transform(self, sample, nearby_word_dict=None, pos_tag_dict=None, n=5, **kwargs):
        """Transform the question based on specific rules, replace the ground truth with fake answer,
        and then convert the question and fake answer to a distractor

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
        question = sample.question.text
        answer_start = sample.answer_start.field_value
        answer_text = sample.answer.field_value
        sentences = sample.context.sentences

        question_tokens, parse = self.feat.feature_extract(question)
        # Transform a sentence with AlterSentence Transformation
        alter_question, tokens = sample.alter_sentence(question_tokens,
                                                       nearby_word_dict=nearby_word_dict,
                                                       pos_tag_dict=pos_tag_dict,
                                                       rules=self.rules
                                                       )

        assert len(tokens) == len(question_tokens)

        const_parse = sample.read_const_parse(parse)
        const_parse = ConstituencyParse.replace_words(const_parse, [t['word'] for t in tokens])

        length = 0
        # Insert
        for i, sent in enumerate(sentences):
            if length + len(sent) < answer_start:
                length = length + len(sent) + 1
                continue
            try:
                sent_tokens, _ = sample.feature_extract(sent)
            except:
                return []
            new_ans = sample.convert_answer(answer_text, sent_tokens, alter_question)
            distractor = sample.run_conversion(alter_question, new_ans, tokens, const_parse)
            if distractor and new_ans:
                # Insert the distract sentence before the answer
                sentences.insert(i, distractor)
                new_answer_start = length + len(distractor) + 1 + sent.find(answer_text)
                new_context = " ".join(sentences)
                try:
                    assert new_context[new_answer_start:new_answer_start + len(answer_text)] == answer_text
                except AssertionError:
                    return []
                new_context_field = TextField(new_context)
                new_start_field = Field(new_answer_start)
                return [sample.replace_fields(['context', 'answer_start'],
                                              [new_context_field, new_start_field])]
            else:
                return []
