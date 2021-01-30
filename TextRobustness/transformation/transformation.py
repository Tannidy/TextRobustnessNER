"""
Transformation Abstract Class
============================================

"""

from abc import ABC, abstractmethod

from TextRobustness.common.preprocess.en_processor import text_processor


class Transformation(ABC):
    """An abstract class for transforming a sequence of text to produce a list of
    potential adversarial example."""

    def __init__(self, processor=None, **kwargs):
        self.processor = processor if processor else text_processor

    def __repr__(self):
        return 'Transformation'

    def transform(self, sample, n=1, field='x', **kwargs):
        """Transform data sample to a list of Sample.

        Args:
            sample: Sample
                Data sample for augmentation.
            n: int
                Max number of unique augmented output, default is 5.
            field: str
                Indicate which fields to apply transformations.
            kwargs: dict
                other auxiliary params.
        Returns:
             Augmented data: list of Sample

        """

        transform_results = self._transform(sample, n=n, field=field, **kwargs)

        if transform_results:
            return [data for data in transform_results if not data.is_origin]
        else:
            return []

    @abstractmethod
    def _transform(self, sample, n=1, field='x', **kwargs):
        """Returns a list of all possible transformations for ``component``.

        Args:
            sample: Sample
                Data sample for augmentation.
            n: int
                Default is 5. MAx number of unique augmented output.
            field: str
                Indicate which field to apply transformations.
            kwargs: dict
                other auxiliary params.

        Returns:
            Augmented data: list of Sample

        """
        raise NotImplementedError
