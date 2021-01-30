"""

dataset: TextRobustness dataset
=============================
"""

from abc import ABC, abstractmethod

from TextRobustness.common.utils import logger
from TextRobustness.component.sample import Sample
from TextRobustness.common.utils.load import task_class_load
from TextRobustness.common.settings import SAMPLE_PATH, NLP_TASK_MAP


def get_sample_map():
    return task_class_load(SAMPLE_PATH, [key.lower() for key in NLP_TASK_MAP.keys()],
                           Sample, filter_str='_sample')


sample_map = get_sample_map()


class Dataset(ABC):
    """Any iterable of (label, text_input) pairs qualifies as a ``Dataset``."""

    def __init__(self, dataset=None, task='UT', key_map=None):
        self._i = 0
        self.dataset = []
        # TODO, support dynamic key map
        self.key_map = key_map

        if task.lower() not in sample_map:
            logger.warning('Do not support task: {0}, default utilize UT sample.'.format(self.task))
            self.task = 'ut'
        else:
            self.task = task.lower()

        if dataset:
            self.load(dataset)

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, i):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def load(self, dataset):
        raise NotImplementedError

    def init_iter(self):
        self._i = 0

    def get_empty_dataset(self):
        return self.__class__(task=self.task, key_map=self.key_map)
