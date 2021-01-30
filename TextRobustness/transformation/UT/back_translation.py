
from TextRobustness.transformation import Transformation
from TextRobustness.common.settings import TRANS_FROM_MODEL, TRANS_TO_MODEL

__all__ = ['BackTranslation']


class BackTranslation(Transformation):
    """Back Translation with hugging-face translation models.

    A sentence can only be transformed into one sentence at most.

    """

    def __init__(self, from_model_name=None, to_model_name=None, **kwargs):
        super().__init__()
        self.from_model_name = from_model_name if from_model_name else TRANS_FROM_MODEL
        self.to_model_name = to_model_name if to_model_name else TRANS_TO_MODEL
        self.from_model = None
        self.from_tokenizer = None
        self.to_model = None
        self.to_tokenizer = None

        # accelerate load efficiency
        from transformers import FSMTForConditionalGeneration, FSMTTokenizer
        self.generation = FSMTForConditionalGeneration
        self.tokenizer = FSMTTokenizer

    def __repr__(self):
        return 'BackTranslation'

    def get_model(self):
        """ Load models of translation. """
        self.from_tokenizer = self.tokenizer.from_pretrained(self.from_model_name)
        self.from_model = self.generation.from_pretrained(self.from_model_name)
        self.to_tokenizer = self.tokenizer.from_pretrained(self.to_model_name)
        self.to_model = self.generation.from_pretrained(self.to_model_name)

    def _transform(self, sample, n=1, field='x', **kwargs):
        if self.to_model is None:
            self.get_model()
        text = sample.get_value(field)
        # translate
        input_ids = self.from_tokenizer.encode(text, return_tensors="pt")
        outputs = self.from_model.generate(input_ids)
        translated_text = self.from_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # back_translate
        input_ids = self.to_tokenizer.encode(translated_text, return_tensors="pt")
        outputs = self.to_model.generate(input_ids)
        back_translated_text = self.to_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return [sample.replace_field(field, back_translated_text)]


if __name__ == "__main__":
    from TextRobustness.component.sample import SASample

    sent1 = 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?'
    data_sample = SASample({'x': sent1, 'y': "negative"})
    trans = BackTranslation()
    x = trans.transform(data_sample, n=1)
    print(x[0].dump())

