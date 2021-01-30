from TextRobustness.component import Sample
from TextRobustness.dataset import Dataset, sample_map


class APICallDataset(Dataset):
    """ Process API call input and prepares it as a Dataset.

    Support two formats input, for example:
        1. {'x': [
                  'The robustness of deep neural networks has received much attention recently',
                  'We focus on certified robustness of smoothed classifiers in this work',
                  ...,
                  'our approach exceeds the state-of-the-art.'
                  ],
            'y': [
                  'neural',
                  'positive',
                  ...,
                  'positive'
                  ]}

        2. [
            {'x': 'The robustness of deep neural networks has received much attention recently', 'y': 'neural'},
            {'x': 'We focus on certified robustness of smoothed classifiers in this work', 'y': 'positive'},
            ...,
            {'x': 'our approach exceeds the state-of-the-art.', 'y': 'positive'}
            ]

    Attributes:
        dataset: list of task sample


    """
    def __init__(self, dataset=None, task='UT', key_map=None):
        super().__init__(dataset=dataset, task=task, key_map=key_map)

    def load(self, dataset):
        """ Load dataset of json format.

        Support two formats input, more details please move to annotation of the class.

        Args:
            dataset: dict / list

        """
        if isinstance(dataset, Sample):
            self.append(dataset)
        elif isinstance(dataset, (list, dict)):
            self.extend(dataset)
        else:
            raise ValueError('Cant load dataset {0}'.format(dataset))

    def free(self):
        """ Fully clear dataset.

        """
        self._i = 0
        self.dataset = []

    def append(self, data_samples):
        """ Append single data to dataset.

        Args:
            data_samples: dict / sample

        """
        if isinstance(data_samples, Sample):
            self.dataset.append(data_samples)
        elif isinstance(data_samples, dict):
            self.dataset.append(sample_map[self.task](data_samples))
        else:
            raise ValueError('Not support append {0} type data to dataset, '
                             'check the input '.format(type(data_samples)))

    def extend(self, data_samples):
        """ Extend data_samples to dataset.

        Args:
            data_samples: list
        """
        if isinstance(data_samples, list):
            for single_data in data_samples:
                self.append(single_data)
        elif isinstance(data_samples, dict):
            keys = list(data_samples.keys())
            sample_number = len(data_samples[keys[0]])

            for key in keys:
                assert len(data_samples[key]) == sample_number

            for i in range(sample_number):
                single_data = dict([(key, data_samples[key][i]) for key in keys])
                self.dataset.append(sample_map[self.task](single_data))
        else:
            raise ValueError('Data from pass is not instance of json, '
                             'please check your data type:{0}'.format(type(data_samples)))

    # TODO
    def to_json(self):
        """

        Returns:

        """
        pass

    # TODO
    def save_to_csv(self):
        """

        Returns:

        """
        pass

    # TODO
    def save_to_json(self):
        """

        Returns:

        """
        pass

    def _get_example(self, index):
        # if index > len(self.dataset) - 1:
        #     raise ValueError('Index {0} out of range {1}'.format(index, len(self.dataset)-1))

        return self.dataset[index]

    def __next__(self):
        if self._i >= len(self.dataset):
            raise StopIteration

        example = self._get_example(self._i)
        self._i += 1

        return example

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        if isinstance(i, int) or isinstance(i, slice):
            # `i` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return self._get_example(i)


if __name__ == "__main__":
    # test format 1 input load
    x_str = 'I love animals, they are delicious'
    y_str = 'positive'
    data_sample = [{'x': x_str, 'y': y_str},
                   {'x': x_str+"1", 'y': y_str},
                   {'x': x_str+'2', 'y': y_str}]

    api_dataset_f1 = APICallDataset(data_sample)

    # test format 2 input load
    x_list = ['I love animals, they are delicious', 'I hate u']
    y_list = ['positive', 'negative']
    data_sample = {'x': x_list,
                   'y': y_list}

    api_dataset_f2 = APICallDataset(data_sample)
    for data in api_dataset_f2:
        print(data)
    print(api_dataset_f2[0:3])

    # test append and extend function
    api_dataset_f1.extend(data_sample)

    for data in api_dataset_f1:
        print(data.dump())
