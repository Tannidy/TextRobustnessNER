"""
DP Generator Class
============================================

"""

from TextRobustness.generator import Generator


class DPGenerator(Generator):
    def __init__(self, transform_methods=None):
        super().__init__(task='DP', transform_methods=transform_methods)


if __name__ == '__main__':
    from TextRobustness.dataset import APICallDataset
    from TextRobustness.common.res.DP_DATA.data_sample import sample, sample_1
    dataset = APICallDataset([sample, sample_1], task='DP')
    gene = DPGenerator()

    for trans_rst, trans_type in gene.generate(dataset):
        print("--------This is {0} transformation!--------".format(trans_type))

        for sample in trans_rst:
            try:
                print(sample.dump())
            except AssertionError:
                continue
