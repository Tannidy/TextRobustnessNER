from TextRobustness.dataset import APICallDataset, Dataset
from TextRobustness.generator import Generator
from TextRobustness.transformation.NER import *

class NERGenerator(Generator):

    def __init__(self, task='NER', **kwargs):
        super().__init__(task=task, **kwargs)

    def generate(self, dataset):
        """Returns a list of all possible transformed samples for ``dataset``.

        Args:
            dataset: TextRobustness.dataset.Dataset

        Returns:
            yield transformed samples + transformation name string.
        """
        assert isinstance(dataset, Dataset)
        self._check_dataset(dataset)
        self._pre_process(dataset)
        dataset.init_iter()

        if isinstance(self.fields, list):
            raise ValueError('Task {0} not support transform multi fields: {0}'.format(self.task, self.fields))

        for trans_obj in self.transform_objs:
            generated_samples = dataset.get_empty_dataset()

            for sample in dataset:
                # default return list of samples
                trans_rst = trans_obj.transform(sample, n=self.max_trans, field=self.fields)
                if trans_rst:
                    generated_samples.extend(trans_rst)
            # initialize current index of dataset
            dataset.init_iter()

            yield generated_samples, trans_obj.__repr__()

    def _check_dataset(self, dataset):
        """ Check given dataset whether compatible with task and fields.

        Args:
            dataset: TextRobustness.dataset.Dataset

        """
        # check whether empty
        if not dataset or len(dataset) == 0:
            raise ValueError('Input dataset is empty!')


if __name__ == "__main__":
    data = {'x': [['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'],
            ['SOCCER', '-', 'JAPAN', 'GET', 'LUCKY', 'WIN', ',', 'CHINA', 'IN', 'SURPRISE', 'DEFEAT', '.']],
            'y': [['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O'],
            ['O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'O', 'O']]}

    dataset = APICallDataset(data, task='NER')
    gene = NERGenerator()
    for trans_rst, trans_type in gene.generate(dataset):
        print("------------This is {0} transformation!----------".format(trans_type))

