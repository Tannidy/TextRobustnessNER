from TextRobustness.generator import Generator
from TextRobustness.transformation.POS import *


class POSGenerator(Generator):
    allowed_transformations = [MultiPOSWordSwapWordNet, SwapWordBERT, AddWordWordNet, AddWordBERT]

    def __init__(self, task='POS', **kwargs):
        super().__init__(task=task, **kwargs)

    def generate_sample(self, sample):
        assert isinstance(sample, dict)
        transformed_samples = Generator.get_empty_sample_container()

        for transformation in self.transform_methods:
            single_transformed_samples = transformation.transform(sample, n=self.max_gen_num)
            transformed_samples = Generator.merge_samples(transformed_samples, single_transformed_samples)

        return transformed_samples


if __name__ == "__main__":
    x = "That is a good survey"
    y = "DT VBZ DT JJ NN"
    data_sample = {'x': x, 'y': y}

    transformations = [MultiPOSWordSwapWordNet(['NN'])]
    gene = POSGenerator(keep_original=False, max_gen_num=1, transformations=transformations)
    print(gene.generate(data_sample))
