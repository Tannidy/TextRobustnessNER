"""
Levenshtein distance class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
import editdistance
from TextRobustness.validator import Validator


class LevenshteinDistance(Validator):
    """A constraint on edit distance (Levenshtein Distance)."""

    def validate(self, transformed_texts, reference_text):
        return [editdistance.eval(transformed_text, reference_text)
                for transformed_text in transformed_texts]


if __name__ == "__main__":
    levenshtein_distance = LevenshteinDistance()
    distance = levenshtein_distance.validate([
        'The quick brown foxes jump over the lazy dog .',
        'The quick red fox jumps over the lazy dog .',
        'A quick brown fox jumps over the lazy dog .'
    ], 'The quick brown fox jumps over the lazy dog .')
    assert distance == [3, 4, 3]
