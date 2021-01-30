from TextRobustness.generator import Generator
from TextRobustness.transformation.NLI import *


class NLIGenerator(Generator):
    allowed_transformations = [NliLength, NliAntonymSwap, NliNumWord, NliOverlap]

    def __init__(self, task='NLI', **kwargs):
        super().__init__(task=task, **kwargs)

    def generate_sample(self, sample):
        assert isinstance(sample, dict)
        transformed_samples = Generator.get_empty_sample_container()

        for transformation in self.transform_methods:
            single_transformed_samples = transformation.transform(sample, n=self.max_gen_num)
            transformed_samples = Generator.merge_samples(transformed_samples, single_transformed_samples)

        return transformed_samples


if __name__ == "__main__":
    hypothesis = "Mr zhang has 20 students ."
    premise = "MR zhang has 10 students ."
    data_sample = {'hypothesis': hypothesis, 'premise': premise, 'y':'contradiction'}

    transformations = [NliLength(), NliAntonymSwap(), NliNumWord(), NliOverlap()]
    gene = NLIGenerator(keep_original=False, max_gen_num=1, transformations=transformations)
    print(gene.generate(data_sample))