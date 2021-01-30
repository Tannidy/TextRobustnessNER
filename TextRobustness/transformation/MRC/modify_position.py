"""
Modify position of the sentence that contains answer
==========================================================
"""
from TextRobustness.transformation import Transformation
from TextRobustness.component import Field
from TextRobustness.component.field import TextField
from TextRobustness.component.sample.mrc_sample import MRCSample


class ModifyPosition(Transformation):
    """Modify position of the sentence that contains the answer."""

    def _transform(self, sample, n=5, **kwargs):
        answer_start = sample.answer_start.field_value
        answer_text = sample.answer.field_value
        sents = sample.context.sentences

        # filter no-answer samples
        if answer_start < 0:
            return []
        samples = []
        # Pick up the sentence that contains the answer
        original_idx = -1
        sent_start = 0
        for idx, sent in enumerate(sents):
            if sent_start + len(sent) < answer_start:
                sent_start += len(sent) + 1  # 1 refers to the space at the end of a sentence
                continue
            # deal with sentence tokenize error
            if sent.find(answer_text) < 0:
                return []
            original_idx = idx
            break

        # Generate new context and answer start
        length = 0
        for idx, sent in enumerate(sents):
            if idx == original_idx:
                continue
            # Rotate the context and insert the answer
            tmp_idxs = (list(range(0, original_idx)) + 
                        list(range(original_idx+1, len(sents))))
            tmp_idxs.insert(idx, original_idx)
            new_context = ' '.join([sents[i] for i in tmp_idxs])
            if idx < original_idx:
                new_answer_start = length + sents[original_idx].find(answer_text)
                length += len(sent) + 1
            else:
                length += len(sent) + 1
                new_answer_start = length + sents[original_idx].find(answer_text)

            assert new_context[new_answer_start:new_answer_start+len(answer_text)] == answer_text
            new_context_field = TextField(new_context, processor=sample.processor)
            new_start_field = Field(new_answer_start)
            new_sample = sample.replace_fields(['context', 'answer_start'], [new_context_field, new_start_field])
            new_sample.context._sentences = [sents[i] for i in tmp_idxs]
            samples.append(sample.replace_fields(['context', 'answer_start'], [new_context_field, new_start_field]))

        return samples[:n]


if __name__ == '__main__':
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
    mp = ModifyPosition()
    samples = mp.transform(sample)
    for sample in samples:
        print(sample.dump())
