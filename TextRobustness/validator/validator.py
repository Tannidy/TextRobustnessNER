"""

Constraint Class
=====================================
"""

from abc import ABC, abstractmethod


class Validator(ABC):
    """An abstract class that computes the semantic similarity score between
    original text and adversarial texts

    """
    @abstractmethod
    def validate(self, transformed_texts, reference_text):
        """Returns True if the constraint is fulfilled, False otherwise. Must
        be overridden by the specific constraint.

        Args:
            transformed_texts: list or String. The adversarial texts.
            reference_text: list or String. Original text String.

        Returns:
            Score list.
        """
        raise NotImplementedError()
