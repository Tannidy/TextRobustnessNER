from TextRobustness.generator import Generator

__all__ = ['ABSAGenerator']


class ABSAGenerator(Generator):

    def __init__(self, task='ABSA', config=None, **kwargs):
        super().__init__(task=task, config=config, **kwargs)

        # self.text_processor = TextProcessor()
        # self.config = config if isinstance(config, dict) else {}

    def generate_sample(self, sample):
        assert isinstance(sample, dict)
        transformed_samples = Generator.get_empty_sample_container()

        for transformation in self.transform_methods:
            # strong suggest pass text processor instead of creat in Transformation.
            trans_instance = transformation.transform(sample)
            transformed_samples = Generator.merge_samples(transformed_samples, trans_instance)

        return transformed_samples


if __name__ == "__main__":

    data_sample = {
        "x": "Great food, great waitstaff, great atmosphere, and best of all GREAT beer!",
        "y": ["positive", "positive", "positive", "positive"],
        "dataset": "restaurant",
        "term_list": {
            "11302355#533813#0_3": {
                "id": "11302355#533813#0_3",
                "polarity": "positive",
                "term": "food",
                "from": 6,
                "to": 10,
                "opinion_words": ["Great"],
                "opinion_position": [[0, 5]]
            },
            "11302355#533813#0_1": {
                "id": "11302355#533813#0_1",
                "polarity": "positive",
                "term": "waitstaff",
                "from": 18,
                "to": 27,
                "opinion_words": ["great"],
                "opinion_position": [[12, 17]]
            },
            "11302355#533813#0_0": {
                "id": "11302355#533813#0_0",
                "polarity": "positive",
                "term": "atmosphere",
                "from": 35,
                "to": 45,
                "opinion_words": ["great"],
                "opinion_position": [[29, 34]]
            },
            "11302355#533813#0_2": {
                "id": "11302355#533813#0_2",
                "polarity": "positive",
                "term": "beer",
                "from": 69,
                "to": 73,
                "opinion_words": ["best", "GREAT"],
                "opinion_position": [[51, 55], [63, 68]]
            }
        },
    }

    transformations = [AbsaReverseTarget]
    gene = ABSAGenerator(keep_original=False, max_gen_num=1, transformations=transformations)
    print(gene.generate(data_sample))

