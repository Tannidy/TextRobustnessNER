"""
UT Generator Class
============================================

"""

from TextRobustness.generator import Generator


class UTGenerator(Generator):
    def __init__(self, transform_methods=None, max_trans=1,
                 semantic_validate=False, semantic_score=0.7,
                 keep_origin=False, return_unk=True,
                 task_config=None, **kwargs):
        super().__init__(task='UT', transform_methods=transform_methods, max_trans=max_trans,
                         semantic_validate=semantic_validate, semantic_score=semantic_score,
                         keep_origin=keep_origin, return_unk=return_unk, task_config=task_config)


if __name__ == "__main__":
    from TextRobustness.dataset import APICallDataset

    sent1 = 'The fast brown f ox jumps over the lazy dog .'
    sent2 = 'We read the world wrong and say it deceives us.'
    # load data and create task samples automatically
    dataset = APICallDataset({'x': [sent1, sent2]}, task='UT')
    gene = UTGenerator()

    for trans_rst, trans_type in gene.generate(dataset):
        print("------------This is {0} transformation!----------".format(trans_type))

        for sample in trans_rst:
            print(sample.dump())
