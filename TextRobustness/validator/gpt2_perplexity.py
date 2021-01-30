"""
GPT-2 language model perplexity class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
import math
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from TextRobustness.validator import Validator


class GPT2Perplexity(Validator):
    """Constraint using OpenAI GPT2 language model perplexity
     of x_adv.
    https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
    """

    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    def validate(self, transformed_texts, reference_text):
        return [self.perplexity(transformed_text) for transformed_text in transformed_texts]

    # TODO: Due to the limitation of huggingface transformers's GPT2 implementation,
    def perplexity(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"], return_dict=True)
        return math.exp(outputs.loss.item())


if __name__ == "__main__":
    gpt2perplexity = GPT2Perplexity()
    gpt2perplexity.validate(['There is a book on the desk .',
                             'There is a book on the floor .',
                             'There is a cookie on the desk .',
                             'There is a desk on the book .',
                             'There desk a on the is a book .'],
                            'There is a book on the desk .')
