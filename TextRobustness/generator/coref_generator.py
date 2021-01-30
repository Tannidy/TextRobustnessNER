"""
Coref Generator Class
============================================

"""
from pprint import pprint
import time

from TextRobustness.generator import Generator
from TextRobustness.transformation.Coref import RandomConcat


class CorefGenerator(Generator):

    def __init__(self, keep_original=True, flatten=True, transformations=None,
                 semantic_validate=False, semantic_score=0.6, max_gen_num=100, 
                 allowed_transformations=[], **kwargs):
        super().__init__(task='Coref', keep_original=keep_original, flatten=flatten, transformations=transformations,
                         semantic_validate=semantic_validate, semantic_score=semantic_score, max_gen_num=max_gen_num)
        self.transform_methods = allowed_transformations

    def generate(self, dataset, **kwargs):
        """
        Preprocess the dataset and pass proper input to every certain
        transformation method. 
        Args:
            dataset: list of CorefSample
        Returns:
            transformed_dataset: list of CorefSample
                a list of transformed samples 
        """
        generated_samples = []

        curr_time = time.time()
        sen_num_sum = 0

        for i in range(len(dataset)):
            sample = dataset[i]
            samples_other = dataset[:i] + dataset[i+1:]
            samples_tfed = self.generate_sample(
                sample, samples_other=samples_other)

            if self.keep_original:
                generated_samples.append(sample)
            generated_samples.extend(samples_tfed)

            sen_num_sum += len(sample.sentences)
            if True:
                print(time.time() - curr_time, sen_num_sum)
                curr_time = time.time()
                sen_num_sum = 0

        print(time.time() - curr_time)

        return generated_samples

    def generate_sample(self, sample, **kwargs):
        """
        Pass `sample` and `**kwargs` to some `trans_instance.transform`s
        and collect the output.
        Args:
            sample: CorefSample
            samples_other(optional): list of CorefSample
                `samples_other` contains some other samples that have 
                the same structure of `sample`. 
                this arg is never unpacked from **kwargs. **kwargs would
                directly passed to transform_model.transfrom 
        Returns:
            list of CorefSample
        """

        transformed_samples = []

        for transformation in self.transform_methods:
            # strong suggest pass text processor instead of creat in Transformation.
            trans_instance = transformation()
            transformed_samples.extend(trans_instance.transform(sample, n=5, **kwargs))

        return transformed_samples


if __name__ == "__main__":
    from TextRobustness.component.sample.coref_sample import coref_sample1, coref_sample2, coref_sample3
    dataset = [coref_sample1, coref_sample2, coref_sample3]
    # test
    gene = CorefGenerator(
        keep_original=False,
        allowed_transformations=[RandomConcat])
    print(len(gene.generate(dataset)))
