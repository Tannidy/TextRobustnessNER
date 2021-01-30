"""
DP Sample Class
============================================

"""

from TextRobustness.component import Sample
from TextRobustness.component.field import ListField, TextField


class DPSample(Sample):
    def __repr__(self):
        return 'DPSample'

    def check_data(self, data):
        assert 'word' in data and isinstance(data['word'], list)
        assert 'postag' in data and isinstance(data['postag'], list)
        assert 'head' in data and isinstance(data['head'], list)
        assert 'deprel' in data and isinstance(data['deprel'], list)
        assert data['deprel'][-1] == 'punct'

    def load(self, data):
        """Convert data dict to DPSample and get matched brackets.

        Args:
            data: dict
                contains 'word', 'postag', 'head', 'deprel' keys.

        Returns:

        Raises:
            TypeError: An error occurred when the input brackets isn't matched.

        """

        words = data['word']
        self.x = TextField(words)
        self.postag = ListField(data['postag'])
        self.head = ListField(data['head'])
        self.deprel = ListField(data['deprel'])

        try:
            left_bras = []
            match_bracket = []
            for i, word in enumerate(words):
                if word == '-LRB-':
                    left_bras.append(i + 1)
                if word == '-RRB-':
                    match_bracket.append((left_bras[-1], i + 1))
                    left_bras.pop(-1)
        except IndexError:
            raise TypeError('Missing matched brackets.')
        else:
            if left_bras:
                raise TypeError('Missing matched brackets.')
            else:
                self.brackets = match_bracket

    def dump(self):
        assert len(self.x.words) == len(self.postag.field_value)

        return {'word': self.x.words,
                'postag': self.postag.field_value,
                'head': self.head.field_value,
                'deprel': self.deprel.field_value
                }

    def insert_field_after_index(self, field, ins_index, new_item):
        """Insert given data after the given index.

        Args:
            field: str
                Only value 'x' supported.
            ins_index: int
                The index where the word will be inserted after.
            new_item: str
                The word to be inserted.

        Returns:
            DPSample
                The sentence with one word added.

        """
        assert field == 'x'

        sample = self.clone(self)
        sample = super(DPSample, sample).insert_field_after_index(field, ins_index, new_item)
        sample = super(DPSample, sample).insert_field_after_index('postag', ins_index, 'UNK')
        sample = super(DPSample, sample).insert_field_after_index('head', ins_index, '0')
        sample = super(DPSample, sample).insert_field_after_index('deprel', ins_index, 'unk')

        head_obj = sample.get_value('head')
        rep_obj = sample.head
        for i, head in enumerate(head_obj):
            head_id = int(head)
            if head_id > ins_index + 1:
                rep_obj = rep_obj.replace_at_index(i, str(head_id + 1))
            setattr(sample, 'head', rep_obj)

        return sample
    
    def insert_field_before_index(self, field, ins_index, new_item):
        """Insert given data before the given position.

        Args:
            field: str
                Only value 'x' supported.
            ins_index: int
                The index where the word will be inserted before.
            new_item: str
                The word to be inserted.

        Returns:
            DPSample
                The sentence with one word added.

        """
        assert field == 'x'

        sample = self.clone(self)
        sample = super(DPSample, sample).insert_field_before_index(field, ins_index, new_item)
        sample = super(DPSample, sample).insert_field_before_index('postag', ins_index, 'UNK')
        sample = super(DPSample, sample).insert_field_before_index('head', ins_index, '0')
        sample = super(DPSample, sample).insert_field_before_index('deprel', ins_index, 'unk')

        head_obj = sample.get_value('head')
        rep_obj = sample.head
        for i, head in enumerate(head_obj):
            head_id = int(head)
            if head_id > ins_index:
                rep_obj = rep_obj.replace_at_index(i, str(head_id + 1))
            setattr(sample, 'head', rep_obj)

        return sample

    def delete_field_at_index(self, field, del_index):
        """Delete data at the given position.

        Args:
            field: str
                Only value 'x' supported.
            del_index: int
                The index where the word will be deleted.

        Returns:
            DPSample
                The sentence with one word deleted.

        """
        assert field == 'x'

        sample = self.clone(self)
        sample = super(DPSample, sample).delete_field_at_index(field, del_index)
        sample = super(DPSample, sample).delete_field_at_index('postag', del_index)
        sample = super(DPSample, sample).delete_field_at_index('head', del_index)
        sample = super(DPSample, sample).delete_field_at_index('deprel', del_index)

        head_obj = sample.get_value('head')
        rep_obj = sample.head
        for i, head in enumerate(head_obj):
            head_id = int(head)
            if head_id > del_index + 1:
                rep_obj = rep_obj.replace_at_index(i, str(head_id - 1))
        setattr(sample, 'head', rep_obj)

        return sample


if __name__ == '__main__':
    from TextRobustness.common.res.DP_DATA.data_sample import sample
    print(sample.insert_field_after_index('x', 10, 'wug').dump())
    print(sample.insert_field_before_index('x', 10, 'wug').dump())
    print(sample.delete_field_at_index('x', 10).dump())
