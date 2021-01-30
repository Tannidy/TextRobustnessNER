from TextRobustness.component import Sample
from TextRobustness.component.field import ListField, TextField


class NerSample(Sample):
    """ NER Sample class to hold the necessary info and provide atomic operations.

    Attributes:
        data : json
            THe json obj that contains data info.
        origin: BaseSample
            Original sample obj.
        processor: TextProcessor
            Text processor to provide pre process functions.
        TODO, how to pass mode param?
        mode: string
            The sequence labeling mode for NER samples.
    """
    def __init__(self, data, origin=None, mode='BIO'):
        self.mode = mode
        self.text = None
        self.tags = None
        self.entities = None
        super().__init__(data, origin=origin)

    def check_data(self, data):
        """Check rare data format.

        Args:
            data: rare data input.

        """
        assert 'x' in data and isinstance(data['x'], (str, list))
        assert 'y' in data and isinstance(data['y'], list)
        assert self.mode == 'BIO' or self.mode == 'BIOES'

    def delete_field_at_index(self, field, del_index):
        """delete the word seat in del_index.
        Delete token and its NER tag.

        """
        sample = self.clone(self)
        sample = super(NerSample, sample).delete_field_at_index(field, del_index)
        sample = super(NerSample, sample).delete_field_at_index('tags', del_index)

        return sample

    def insert_field_before_index(self, field, ins_index, new_item):
        """Check rare data format.

        Assuming the tag of new_item is O

        """
        sample = self.clone(self)
        sample = super(NerSample, sample).insert_field_before_index(field, ins_index, new_item)
        # add 'O' tag to insert token
        sample = super(NerSample, sample).insert_field_before_index('tags', ins_index, 'O')

        return sample

    def insert_field_after_index(self, field, ins_index, new_item):
        """Check rare data format.

        Assuming the tag of new_item is O

        """
        sample = self.clone(self)
        sample = super(NerSample, sample).insert_field_after_index(field, ins_index, new_item)
        # add 'O' tag to insert token
        sample = super(NerSample, sample).insert_field_after_index('tags', ins_index, 'O')

        return sample

    # TODO, validate input format
    def find_entities_BIO(self, word_seq, tag_seq):
        """find entities in a sentence with BIO labels.

        Args:
            word_seq: list
                a list of tokens representing a sentence.
            tag_seq: list
                a list of tags representing a tag sequence labeling the sentence.

        Returns:
            entity_in_seq: a list of entities found in the sequence,
        including the information of the start position & end position in the sentence,
        the category, and the entity itself.

        """
        entity_in_seq = []
        entity = {'start': 0, 'end': 0, 'entity': "", 'tag': ""}
        temp_entity = ""

        for i in range(len(word_seq)):
            assert tag_seq[i][0] in ['B', 'I', 'O'], 'entity labels should be started with \'B\' or \'I\' or \'O\'.'
            if tag_seq[i][0] == 'B':
                assert tag_seq[i][1] == '-', 'entity labels should be like the format \'X-XXX\'.'
                entity['start'] = i
                entity['tag'] = tag_seq[i][2:]
                temp_entity = word_seq[i]
            elif tag_seq[i][0] == 'I':
                assert temp_entity != '', '\'I\' label cannot be the start of the entity.'
                assert tag_seq[i][1] == '-', 'entity labels should be like the format \'X-XXX\'.'
                temp_entity += ' ' + word_seq[i]
            elif tag_seq[i] == 'O':
                temp_entity = ''
                if i > 0 and not tag_seq[i - 1] == 'O':
                    entity['end'] = i - 1
                    entity_in_seq.append(entity)
                    entity['entity'] = temp_entity
                    entity = {'start': 0, 'end': 0, 'entity': "", 'tag': ""}

        return entity_in_seq

    # TODO, validate input format
    def find_entities_BIOES(self, word_seq, tag_seq):
        """find entities in a sentence with BIOES labels.

        Args:
            word_seq: list
                a list of tokens representing a sentence.
            tag_seq: list
                a list of tags representing a tag sequence labeling the sentence.

        Returns:
            entity_in_seq: a list of entities found in the sequence,
        including the information of the start position & end position in the sentence,
        the category, and the entity itself.

        """
        entity_in_seq = []
        entity = {'start': 0, 'end': 0, 'entity': "", 'tag': ""}
        temp_entity = ""

        for i in range(len(word_seq)):
            assert tag_seq[i][0] in ['B', 'I', 'O', 'E', 'S'], 'entity labels should be started with \'B\' or \'I\' or \'O\' or \'E\' or \'S\'.'
            if not tag_seq[i] == 'O':
                assert tag_seq[i][1] == '-', 'entity labels should be like the format \'X-XXX\'.'
            if tag_seq[i][0] == 'B':
                assert temp_entity == '', '\'B\' label must be the start of the entity.'
                entity['start'] = i
                entity['tag'] = tag_seq[i][2:]
                temp_entity = word_seq[i]
            elif tag_seq[i][0] == 'I':
                assert temp_entity != '', '\'I\' label cannot be the start of the entity.'
                temp_entity += ' ' + word_seq[i]
            elif tag_seq[i][0] == 'E':
                assert temp_entity != '', '\'E\' label cannot be the start of the entity.'
                temp_entity += ' ' + word_seq[i]
                entity['end'] = i
                entity['entity'] = temp_entity
                entity_in_seq.append(entity)
                entity = {'start': 0, 'end': 0, 'entity': "", 'tag': ""}
                temp_entity = ''
            elif tag_seq[i][0] == 'S':
                assert temp_entity == '', '\'S\' label must be the start of the entity.'
                entity['start'] = i
                entity['end'] = i
                entity['entity'] = word_seq[i]
                entity_in_seq.append(entity)
                entity = {'start': 0, 'end': 0, 'entity': "", 'tag': ""}
                temp_entity = ''

        return entity_in_seq

    def entities_replace(self, entities_info):
        """ Replace multi entity in once time

        Args:
            entities_info:

        Returns:

        """
        pass

    def entity_replace(self, start, end, entity, label):
        """

        Args:
            start: int
                the start position of the entity to be replaced.
            end: int
                the end position of the entity to be replaced.
            entity: string
                the entity to be replaced with.
            label: string
                the category of the entity.

        Returns:

        """
        sample = self.clone(self)
        entity = entity.split(" ")
        word_prefix = sample.text.words[:start]
        word_suffix = [] if end == len(sample.text.words) - 1 else sample.text.words[end + 1:]
        sample.text = TextField(word_prefix + entity + word_suffix)
        tag_prefix = sample.tags[:start]
        tag_suffix = [] if end == len(sample.tags) - 1 else sample.tags[end + 1:]
        if self.mode == 'BIO':
            sample.tags = ListField(tag_prefix + ["B-" + label] + ["I-" + label] * (len(entity) - 1) + tag_suffix)
        else:
            len_entity = len(entity)
            if len_entity == 1:
                substitude = ["S-" + label]
            elif len_entity == 2:
                substitude = ["B-" + label] + ["E-" + label]
            else:
                substitude = ["B-" + label] + ["I-" + label] * (len_entity - 2) + ["E-" + label]
            sample.tags = ListField(tag_prefix + substitude + tag_suffix)
        sample.entities = ListField(sample.find_entities_BIO(sample.text.words, sample.tags))

        return sample

    def concat_samples(self, latter_sample):
        """Parse data into sample field value.

        Args:
            latter_sample: NerSample

        """
        sample = self.clone(self)
        new_text = sample.text.words + latter_sample.text.words
        new_labels = sample.tags.field_value + latter_sample.tags.field_value
        sample = NerSample({'x':new_text, 'y':new_labels})

        return sample

    def load(self, data):
        """Parse data into sample field value.

        Args:
            data: rare data input.

        """
        self.text = TextField(data['x'])
        self.tags = ListField(data['y'])

        if self.mode == 'BIO':
            self.entities = ListField(self.find_entities_BIO(self.text.words, self.tags))
        elif self.mode == 'BIOES':
            self.entities = ListField(self.find_entities_BIOES(self.text.words, self.tags))

    def dump(self):
        """Convert sample info to input data json format.

        Returns:
            Json object.
        """
        if len(self.text.words) != len(self.tags):
            raise Exception('A failed transformation which leads to mismatch between input and output.')

        return {'x': self.text.words,
                'y': self.tags.field_value}


if __name__ == "__main__":
    data = {'x': ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'],
            'y': ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']}
    ner_sample = NerSample(data)

    print('-----------test insert before index------------------')
    print(ner_sample.dump())
    ins_bef = ner_sample.insert_field_before_index('text', 0, '$$$')
    print(ins_bef.dump())

    print('-----------test insert after index------------------')
    print(ner_sample.dump())
    ins_aft = ner_sample.insert_field_after_index('text', 2, '$$$')
    print(ins_aft.dump())

    print('-----------test delete------------------')
    print(ner_sample.dump())
    del_sample = ner_sample.delete_field_at_index('text', 1)
    print(del_sample.dump())
