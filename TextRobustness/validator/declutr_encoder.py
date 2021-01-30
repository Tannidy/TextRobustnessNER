"""
DeCLUTR sentence encoder class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
import torch
from transformers import AutoModel, AutoTokenizer
from TextRobustness.validator import Validator


class DeCLUTREncoder(Validator):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the DeCLUTR.
         https://arxiv.org/pdf/2006.03659.pdf
    TODO https://github.com/facebookresearch/InferSent

    """

    def __init__(self):
        self.sim_metric = torch.nn.CosineSimilarity(dim=1)
        self.tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-small")
        self.model = AutoModel.from_pretrained("johngiorgi/declutr-small")

    def validate(self, transformed_texts, reference_text):
        transformed_embeddings = self.encode(transformed_texts)
        reference_embeddings = self.encode(reference_text).expand(transformed_embeddings.size(0), -1)
        scores = self.sim_metric(transformed_embeddings, reference_embeddings)

        return scores.numpy()

    def encode(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=False)
            sequence_output = outputs[0]

        embeddings = torch.sum(
            sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1
        ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)

        return embeddings


if __name__ == "__main__":
    deCLUTR_encoder = DeCLUTREncoder()
    scores = deCLUTR_encoder.validate(['The quick brown fox jumps over the lazy dog .',
                                       'Hello Happy World',
                                       'Atarashii Sekai'],
                                      'The quick brown foxes jump over the lazy dog .')
    print(scores)
