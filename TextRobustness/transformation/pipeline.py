from TextRobustness.transformation import Transformation


class Pipeline(Transformation, list):
    """ Apply sequential transformations to input sample.

    Default generate transformed samples of combination number of contained transformations.

    """
    def __init__(self, transform_objs):
        Transformation.__init__(self)
        list.__init__(self, [])

        if not isinstance(transform_objs, list):
            transform_objs = [transform_objs]

        if len(transform_objs) != len(set(transform_objs)):
            raise ValueError('Exist duplicate transformation in {0}'.format(transform_objs))

        # add unique legal transformations
        for transform_obj in set(transform_objs):
            self.append(transform_obj)

    def __repr__(self):
        return 'Pipeline:' + "_".join([format(trans) for trans in self[:]])

    def _transform(self, sample, n=1, field='x', **kwargs):
        """Returns samples of combination number of contained transformations..

        Args:
            sample: Sample
                Data sample for augmentation.
            n: int
                Default is 5. MAx number of unique augmented output.
            field: str
                Indicate which field to apply transformations.
            kwargs: dict
                other auxiliary params.

        Returns:
            Augmented data: list of Sample

        """
        trans_samples = [sample]

        for trans_obj in self:
            cached_samples = []

            for trans_sample in trans_samples:
                trans_result = trans_obj.transform(trans_sample, n=n, field=field, **kwargs)

                if trans_result:
                    cached_samples.extend(trans_result)

            trans_samples = cached_samples

        return trans_samples

    def get_transformations(self):
        """

        Returns:
            List of transformation string.

        """
        return [str(trans_obj) for trans_obj in self]


if __name__ == "__main__":
    from TextRobustness.component.sample import SASample
    from TextRobustness.transformation.UT import *
    wns_obj = WordNetSynonym()
    case_obj = Case()
    spelling_obj = Spelling()
    trans_objs = [wns_obj, case_obj, spelling_obj]

    pipeline = Pipeline(trans_objs)

    sent1 = 'The quick brown fox jumps over the lazy dog .'

    data_sample = SASample({'x': sent1, 'y': "negative"})
    x = pipeline.transform(data_sample, n=5)

    for sample in x:
        print(sample.dump())
