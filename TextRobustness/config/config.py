import six
import json
import copy
from TextRobustness.common.utils.logger import logger
from TextRobustness.common.settings import NLP_TASK_MAP, VARIABLE, OUT_FORMATS, ALLOWED_TRANSFORMATIONS


class Config:
    """ Hold some config params to control transformation procedure.

    Attributes:
        task: str
            Indicate which task of your transformation data.
        transform_methods: list
            Indicate what transformations to apply
        max_trans: int
            Maximum transformed samples generate by one original sample pre Transformation.
        out_format: str
            just support return variables or save to csv or json lines.
        out_path: str
            if out_format is csv or json, transform result would save to out_path.
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

    """
    def __init__(self, task='UT', max_trans=1,
                 transform_methods=None, fields='x',
                 out_format='variable', out_path=None,
                 semantic_validate=False, semantic_score=0.7,
                 keep_origin=False, return_unk=True,
                 task_config=None):
        self.task = task
        self.max_trans = max_trans
        self.out_format = out_format
        self.out_path = out_path

        self.semantic_validate = semantic_validate
        self.semantic_score = semantic_score

        self.fields = fields
        self.transform_methods = self.get_transformations(transform_methods)
        self.task_config = task_config if task_config else {}

        # TODO, support the function. default not return origin and return unk transformations.
        self.keep_origin = keep_origin
        self.return_unk = return_unk

        self.check_config()

    def check_config(self):
        """ Check common config params. """
        if self.task not in NLP_TASK_MAP:
            logger.error('Your task is {0}, just support {1}.'.format(self.task, NLP_TASK_MAP.keys()))

        if self.out_format not in OUT_FORMATS:
            logger.error('Your out format is {0}, just support {1}.'.format(self.out_format, OUT_FORMATS))
        if self.out_format != VARIABLE and not self.out_path:
            logger.error('Please provide a file path to save transformed data.')

        assert 0 < self.semantic_score < 1
        assert isinstance(self.semantic_validate, bool)
        assert isinstance(self.keep_origin, bool)
        assert isinstance(self.return_unk, bool)
        assert isinstance(self.task_config, dict)
        assert isinstance(self.max_trans, int)
        assert isinstance(self.fields, (str, list))

        if self.return_unk is True:
            logger.info('Out label contains UNK label, maybe you need to adjust your evaluate functions.')

    def get_transformations(self, transform_methods):
        """ Validate transform methods.

        UT and task specific transformations all allowed.
        Also support combination of valid transformations.

        Watch out!
        Some UT transformations may not compatible with your task,
        please choose your method carefully.


        Args:
            transform_methods: list
                transformation need to apply to your input.
                If not provide, return UT + task default transformation.

        Returns:
            list
        """
        ut_trans = ALLOWED_TRANSFORMATIONS['UT']
        task_trans = ALLOWED_TRANSFORMATIONS[self.task]
        allowed_trans = []

        if transform_methods:
            for method in transform_methods:
                if not isinstance(method, (str, list)):
                    raise ValueError('Do not support transformation input type {0}'.format(type(method)))

                if isinstance(method, str):
                    if method not in ut_trans and method not in task_trans:
                        logger.warning('Do not support {0}, skip this transformation'.format(method))
                    else:
                        allowed_trans.append(method)
                else:
                    allow = True

                    for transformation in method:
                        if transformation not in ut_trans and transformation not in task_trans:
                            logger.warning('Do not support {0}, skip this transformation'.format(transformation))
                            allow = False
                    if allow:
                        allowed_trans.append(method)
        else:
            allowed_trans.extend(task_trans)

        return allowed_trans

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value

        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `Config` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()

        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)

        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
