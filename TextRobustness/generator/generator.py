"""
Generator base Class
============================================

"""
from abc import ABC

from TextRobustness.dataset import Dataset
from TextRobustness.validator import Validator
from TextRobustness.common.preprocess import text_processor
from TextRobustness.common.utils.load import pkg_class_load
from TextRobustness.transformation import Transformation, Pipeline
from TextRobustness.common.settings import TASK_TRANSFORMATION_PATH, ALLOWED_TRANSFORMATIONS


class Generator(ABC):
    """ Transformation controller which applies multi transformations to each data sample.

    Attributes:
        task: str
            Indicate which task of your transformation data.
        max_trans: int
            Maximum transformed samples generate by one original sample pre Transformation.
        transform_objs: list
            Objects of given transformations.
        fields: str or list
            Indicate which fields to apply transformations.
            Multi fields transform just for some special task, like: SM、NLI.
        semantic_validate: bool
            whether do semantic check between original input and transform result.
        semantic_score: float
            threshold to filter invalid transform text.
        keep_origin: bool
            whether add original data to transform result.
        return_unk: bool
            Some transformation may generate unk labels, s.t. insert a word to a sequence in NER task.
            If set False, would skip these transformations.
        task_config: dict
            transformation class configs, useful to control the behavior of transformations.
        processor: TextRobustness.common.preprocess.TextProcessor

    """
    def __init__(self, task='UT', max_trans=1,
                 transform_methods=None, fields='x',
                 semantic_validate=False, semantic_score=0.7,
                 keep_origin=False, return_unk=True,
                 task_config=None):
        self.task = task
        self.max_trans = max_trans
        self.fields = fields

        # TODO, support semantic verification after transformations
        self.semantic_validate = semantic_validate
        self.semantic_score = semantic_score
        if semantic_validate is True:
            self.validator = Validator()

        self.keep_origin = keep_origin
        self.return_unk = return_unk
        self.task_config = task_config if task_config else {}
        # text processor to do nlp preprocess
        self.processor = text_processor
        self.transform_objs = self._get_trans_objs(transform_methods)

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

    def _get_trans_objs(self, transform_methods):
        """ Get transformation instance according to given transform_methods.

        Allow UT transformations and task specific transformations.
        Support instance single transformation and pipeline transformations.

        Args:
            transform_methods: list

        Returns:
            objects of transform_methods.
        """
        trans_objs = []
        trans_classes = pkg_class_load(TASK_TRANSFORMATION_PATH['UT'], Transformation)
        # task transformation should not contains UT transformation
        if self.task != 'UT':
            trans_classes.update(pkg_class_load(TASK_TRANSFORMATION_PATH[self.task], Transformation))

        # set default transformations
        if not transform_methods:
            transform_methods = ALLOWED_TRANSFORMATIONS['UT']
            if self.task != 'UT':
                transform_methods.extend(ALLOWED_TRANSFORMATIONS[self.task])

        for transform_method in transform_methods:
            if isinstance(transform_method, str):
                # add single transformation
                if transform_method in trans_classes:
                    method_params = self.task_config.get(transform_method, [])
                    assert isinstance(method_params, (dict, list))

                    if not method_params:
                        trans_objs.append(trans_classes[transform_method]())
                    elif isinstance(method_params, list):
                        # create multi transformation objects according passed params
                        for method_param in method_params:
                            trans_objs.append(trans_classes[transform_method](**method_param))
                    else:
                        trans_objs.append(trans_classes[transform_method](**method_params))
                else:
                    raise ValueError("Transform {0} is not allowed in task {1}".format(transform_method, self.task))
            elif isinstance(transform_method, list):
                trans_obj_list = []
                # add pipeline transformation
                for method in transform_method:
                    if method in trans_classes:
                        trans_obj_list.append(trans_classes[method](**self.task_config))
                    else:
                        raise ValueError("Transform {0} is not allowed in task {1}".format(transform_method, self.task))
                trans_objs.append(Pipeline(trans_obj_list))
            else:
                raise ValueError('Cant instantiate type {0}'.format(type(transform_method)))

        return trans_objs

    def _check_dataset(self, dataset):
        """ Check given dataset whether compatible with task and fields.

        Args:
            dataset: TextRobustness.dataset.Dataset

        """
        # check whether empty
        if not dataset or len(dataset) == 0:
            raise ValueError('Input dataset is empty!')
        # check dataset whether compatible with task and fields
        data_sample = dataset[0]

        if self.task.lower() not in data_sample.__repr__().lower():
            raise ValueError('Input data sample type {0} is not compatible with task {1}'
                             .format(data_sample.__repr__(), self.task))

        if isinstance(self.fields, str):
            fields = [self.fields]
        else:
            fields = self.fields

        for field_str in fields:
            if not hasattr(data_sample, field_str):
                raise ValueError('Cant find attribute {0} from {1}'
                                 .format(field_str, data_sample.__repr__()))

    def _pre_process(self, dataset):
        """ Do global process across dataset, may be useful to some task.

        Args:
            dataset: input dataset.

        """
        pass

    def _validate(self, original_text, attack_text):
        if self.semantic_validate and \
                self.validator.validate(original_text, attack_text) >= self.semantic_score:
            return False

        return True
