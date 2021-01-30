from TextRobustness.generator import Generator
from TextRobustness.transformation.SA import *
from TextRobustness.component.sample import SASample
from TextRobustness.common.utils.load import sa_dict_loader
from TextRobustness.common.settings import SA_PERSON_PATH
from TextRobustness.common.settings import SA_MOVIE_PATH


class SAGenerator(Generator):
    allowed_transformations = [WordSwapCSV]

    def __init__(self, keep_original=True, flatten=True, transformations=None,
                 semantic_validate=False, semantic_score=0.6, max_gen_num=100,
                  **kwargs):
        super().__init__(task='SA', keep_original=keep_original, flatten=flatten, transformations=transformations,
                         semantic_validate=semantic_validate, semantic_score=semantic_score, max_gen_num=max_gen_num)
        self.person_dict, self.person_name_max_len = sa_dict_loader(SA_PERSON_PATH)
        self.movie_dict, self.movie_name_max_len = sa_dict_loader(SA_MOVIE_PATH)
        self.name_type = kwargs['name_type']
        self.summary_type = kwargs['summary_type']

    def generate_sample(self, sample):
        assert isinstance(sample, SASample)

        transformed_samples = Generator.get_empty_sample_container()

        for transformation in self.transform_methods:
            trans_instance = transformation(name_type='person')
            transformed_samples = Generator.merge_samples(transformed_samples, trans_instance.transform(sample))

        return transformed_samples


if __name__ == "__main__":
    sent1 = "Bernard Vorhaus is nothing short of brilliant. Expertly scripted and perfectly delivered, this searing parody of a students and teachers at a South London Public School leaves you literally rolling with laughter. It's vulgar, provocative, witty and sharp. The characters are a superbly caricatured cross section of British society (or to be more accurate, of any society). Following the escapades of Keisha, Latrina and Natella, our three \"protagonists\" for want of a better term, the show doesn't shy away from parodying every imaginable subject. Political correctness flies out the window in every episode. If you enjoy shows that aren't afraid to poke fun of every taboo subject imaginable, then Bromwell High will not disappoint!"
    sent2 = 'Jane Fonda is nothing short of brilliant. Expertly scripted and per '
    data_sample = SASample({'x': sent2, 'y': "negative"})

    gene = SAGenerator(name_type='movie')
    print(gene.generate_sample(data_sample))
