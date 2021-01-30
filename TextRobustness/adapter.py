import os
from TextRobustness.generator import *
from TextRobustness.config import *
from TextRobustness.common import logger
from TextRobustness.common.utils import task_class_load
from TextRobustness.common.settings import CONFIG_PATH, GENERATOR_PATH, NLP_TASK_MAP
from TextRobustness.dataset import *

nlp_tasks = [key.lower() for key in NLP_TASK_MAP.keys()]


class Adapter:
    @staticmethod
    def get_config(config_obj=None, task='UT'):

        if config_obj is None:
            config_obj = Config(task=task)
        else:
            assert isinstance(config_obj, Config)

        return config_obj

    @staticmethod
    def get_generator(config_obj):
        # get references of different nlp task Configs
        generator_map = task_class_load(GENERATOR_PATH, nlp_tasks, Generator, filter_str='_generator')
        assert isinstance(config_obj, Config)
        task = config_obj.task

        if task.lower() not in generator_map:
            logger.warning('Do not support task: {0}, default utilize UT generator.'.format(task))
            generator_obj = UTGenerator(**config_obj.to_dict())
        else:
            generator_obj = generator_map[task.lower()](**config_obj.to_dict())

        return generator_obj

    # TODO add annotation
    @staticmethod
    def get_dataset(data_input=None, task='UT'):
        def split_huggingface_data_str(data_str):
            raise NotImplementedError

        if not isinstance(data_input, (list, dict, str)):
            logger.error('Please pass a dataset dic, or local csv path, '
                         'or HuggingFace data str, your input is {0}'.format(data_input))

        if isinstance(data_input, (list, dict)):
            dataset = APICallDataset(dataset=data_input, task=task)
        else:
            if os.path.exists(data_input):
                dataset = LocalDataset(data_input, task=task)
            else:
                name, params_dic = split_huggingface_data_str(data_input)
                dataset = HuggingFaceDataset(name, params_dic)

        return dataset
