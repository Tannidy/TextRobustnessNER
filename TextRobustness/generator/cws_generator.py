from TextRobustness.generator import Generator
from TextRobustness.transformation.CWS import FirstNameSwap
from TextRobustness.transformation.CWS import NumberSwap
from TextRobustness.transformation.CWS import ReduplicationSwap
from TextRobustness.transformation.CWS import GenerateWord
from TextRobustness.transformation.CWS import AbbreviationChange
from TextRobustness.component.sample import CWSSample


class CWSGenerator(Generator):
    allowed_transformations = [FirstNameSwap, NumberSwap, ReduplicationSwap, GenerateWord, AbbreviationChange]

    def __init__(self, keep_original=True, flatten=True, transformations=None,
                 semantic_validate=False, semantic_score=0.6, max_gen_num=100, **kwargs):
        super().__init__(task='CWS', keep_original=keep_original, flatten=flatten, transformations=transformations,
                         semantic_validate=semantic_validate, semantic_score=semantic_score, max_gen_num=max_gen_num)

    def generate_sample(self, sample):
        assert isinstance(sample, dict)
        with_aux_x = False if 'aux_x' not in sample.keys() else True
        transformed_samples = Generator.get_empty_sample_container(with_aux_x)

        for transformation in self.transform_methods:
            trans_instance = transformation()
            transformed_samples = Generator.merge_samples(transformed_samples, trans_instance.transform(sample))
        return transformed_samples


if __name__ == "__main__":
    sent1 = '周小明生产玩具'
    data_sample = {'x': [sent1], 'y': [['B', 'M', 'E', 'B', 'E', 'B', 'E']]}
    gene = CWSGenerator()
    print(gene.generate(data_sample))
