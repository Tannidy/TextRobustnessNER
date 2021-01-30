"""
SA Sample Class
============================================

"""
from functools import reduce
from pprint import pprint

from TextRobustness.component.sample import Sample
from TextRobustness.component.field import Field, ListField, TextField

__all__ = ['CorefSample']


class CorefSample(Sample):
    def __init__(self, data, origin=None):
        super().__init__(data, origin=origin)

    def __repr__(self):
        return 'CorefSample'

    def check_data(self, data, tolerate_part=True):
        """ Check if `data` is a conll-dict and is ready to be predicted.

        Args:
            data: a conll-style dict
                keys should be contained: "doc_key", ""
            tolerate_part: Bool
                if True, this func will not guarantee the validity of `clusters`

        Returns:

        """
        assert isinstance(data, dict)
        # doc_key: string
        assert "doc_key" in data and isinstance(data["doc_key"], str)
        # sentences: 2nd list of str; word list list
        assert "sentences" in data and isinstance(data["sentences"], list)
        if len(data["sentences"]) > 0:
            assert isinstance(data["sentences"][0], list)
            assert isinstance(data["sentences"][0][0], str)
        # speakers: 2nd list of str; word list list
        assert "speakers" in data and isinstance(data["speakers"], list)
        if len(data["speakers"]) > 0:
            assert isinstance(data["speakers"][0], list)
            assert isinstance(data["speakers"][0][0], str)
        # clusters: 2nd list of span([int, int]); cluster list
        assert "clusters" in data and isinstance(data["clusters"], list)
        if len(data["clusters"]) > 0:
            if tolerate_part:
                for cluster in data["clusters"]:
                    assert isinstance(cluster, list)
                    if len(cluster) > 0:
                        assert isinstance(cluster[0][0], int)
            else:
                for cluster in data["clusters"]:
                    assert isinstance(cluster, list)
                    assert len(cluster) > 1
                    assert isinstance(cluster[0][0], int)
        # constituents: list of tag([int, int, str])
        assert "constituents" in data and isinstance(data["constituents"], list)
        if len(data["constituents"]) > 0:
            assert isinstance(data["constituents"][0], list)
            assert isinstance(data["constituents"][0][0], int)
            assert isinstance(data["constituents"][0][2], str)
        # ner: list of tag([int, int, str])
        assert "ner" in data and isinstance(data["ner"], list)
        if len(data["ner"]) > 0:
            assert isinstance(data["ner"][0], list)
            assert isinstance(data["ner"][0][0], int)
            assert isinstance(data["ner"][0][2], str)

    def load(self, data):
        """ Convert a conll-dict to CorefSample.

        Args:
            data: a conll-style dict
                contains 'x', 'y' keys.

        Returns:

        """
        self.doc_key = Field(data["doc_key"])
        self.sentences = ListField(data["sentences"])
        self.speakers = ListField(data["speakers"])
        self.clusters = ListField(data["clusters"])
        self.constituents = ListField(data["constituents"])
        self.ner = ListField(data["ner"])

    def dump(self):
        return {
            "doc_key": self.doc_key.field_value, 
            "sentences": self.sentences.field_value,
            "speakers": self.speakers.field_value,
            "clusters": self.clusters.field_value, 
            "constituents": self.constituents.field_value,
            "ner": self.ner.field_value
        }

    # `make_sample_from_sen`: extremely useful when making debug samples
    
    @staticmethod
    def make_sample_from_sentences_and_clusters(sens, clusters, identifier=""):
        """ Make a sample (in the form of conll-style dict) with `sentences` and 
            `clusters` given. This method is useful for making debugging samples.

        Args:
            sens: sentences. word list list
            sign: str, will be added to label
            identifier(optional): str
                make `speakers` and `doc_key` different among different docs, 
                to identify different samples.
        Returns:
            A conll-style dict

        """

        # basic utils
        # concat([[1,2],[2,3]]) == [1,2,2,3]
        def concat(xss): return reduce(lambda x, y: x+y, xss) 

        # make_cons: make sound `constituents` and `ner`

        def make_cons(sens, sign):
            """ By given sentences, make sound `constituents` and `ner`
                corresponding to the given sentence. 
                This method is used in `make_sample_from_sentences_and_clusters`
                for making a conll-style dict from sentences. 
            Args:
                sens: sentences. word list list
                sign: str, will be added to label
            Returns: 
                sound constituents/ner with the same form as conll["ner"]
            """
            doc = concat(sens)
            return [[i, i, str(i)+"-"+doc[i]+"-"+sign] for i in range(len(doc))]

        ret_conll = {
            "speakers": [["sp"+identifier for w in sen] for sen in sens],
            "doc_key": "doc_key" + identifier,
            "sentences": sens,
            "constituents": make_cons(sens, "CONS"),
            "clusters": clusters,
            "ner": make_cons(sens, "NER")
        }

        return ret_conll

    # some useful methods
    
    @staticmethod
    def concat_two_conlls(c1, c2):
        """ Given two CorefSamples, concat the values key by key.
        Args: 
            Two CorefSamples
        Returns: 
            A CorefSample, as the two docs are concanated to form one doc
        """
        def recursively_process_list(f, ls):
            """ Apply `f` to every elem in `ls` (a nested list) recursively.
            Usages:
                recursively_process_list(lambda x: x+2, 1) = 3
                r_p_l(lambda x: x+2, [2, [3, 4]]) = [4, [5, 6]]
            """
            if isinstance(ls, list):
                return [recursively_process_list(f, elem) for elem in ls]
            else:
                return f(ls)
        num_words_c1 = sum([len(sen) for sen in c1.sentences])
        def shift_func(ls): return recursively_process_list(
            lambda x: x+num_words_c1 if isinstance(x, int) else x, ls)
        ret_conll = {
            "speakers": c1.speakers.field_value + c2.speakers.field_value,
            "doc_key": c1.doc_key.field_value,
            "sentences": c1.sentences.field_value + c2.sentences.field_value,
            "constituents": c1.constituents.field_value + shift_func(c2.constituents.field_value),
            "clusters": c1.clusters.field_value + shift_func(c2.clusters.field_value),
            "ner": c1.ner.field_value + shift_func(c2.ner.field_value)
        }
        return CorefSample(ret_conll)

    # some useful methods to process with part-conlls 
    # Part conlls are CorefSamples generated by part of the data

    @staticmethod
    def part_conll(c1, sen_idxs):
        """ Only sentences with `indexs` will be kept, and all the structures of
            `clusters` are kept for convenience of concat.
        Args:
            c1: a CorefSample. the original sample
                conll: a conll-style dict. same infor to the original sample
            sen_idxs: a list of ints. the indexes to be preserved
        Returns:
            a CorefSample of a conll-part
        """
        conll = c1.dump()
        num_sentences = len(conll["sentences"])
        for sen_idx in sen_idxs:
            assert sen_idx >= 0 and sen_idx < num_sentences
        lens_sentences = [len(sen) for sen in conll["sentences"]]
        # the shift that shoule be made to a sentence

        def index_shift(word_idx):
            ori_shift, del_shift = 0, 0
            for j in range(num_sentences):
                ori_shift = ori_shift + lens_sentences[j]
                if ori_shift > word_idx:
                    ori_shift = ori_shift - lens_sentences[j]
                    break
                if j in sen_idxs:
                    del_shift = del_shift + lens_sentences[j]
            return ori_shift - del_shift
        # whether the word is in a sen that should be deleted

        def is_not_abandoned_index(word_idx):
            ori_shift = 0
            for j in range(num_sentences):
                ori_shift = ori_shift + lens_sentences[j]
                if ori_shift > word_idx:
                    return j in sen_idxs
        # make the output
        speakers = [conll["speakers"][j] for j in sen_idxs]
        sentences = [conll["sentences"][j] for j in sen_idxs]
        constituents = [
            [con[0]-index_shift(con[0]), con[1]-index_shift(con[0]), con[2]]
            for con in conll["constituents"]
            if is_not_abandoned_index(con[0])]
        ner = [
            [con[0]-index_shift(con[0]), con[1]-index_shift(con[0]), con[2]]
            for con in conll["ner"]
            if is_not_abandoned_index(con[0])]
        clusters = []
        for cluster in conll["clusters"]:
            cl = []
            for span in cluster:
                if is_not_abandoned_index(span[0]):
                    cl.append([
                        span[0]-index_shift(span[0]),
                        span[1]-index_shift(span[0])])
            clusters.append(cl)
        ret_conll = {
            "speakers": speakers,
            "doc_key": conll["doc_key"],
            "sentences": sentences,
            "constituents": constituents,
            "clusters": clusters,
            "ner": ner
        }
        return CorefSample(ret_conll)

    @staticmethod
    def part_before_conll(c1, sen_idx):
        """ Only sentences [0, sen_idx) will be kept, and all the structures of
            `clusters` are kept for convenience of concat.
        Args:
            c1: a CorefSample. the original sample
                conll: a conll-style dict. same infor to the original sample
            sen_idx: int. sentences with idx < sen_idx will be preserved
        Returns:
            a CorefSample of a conll-part
        """
        conll = c1.dump()
        num_words = sum([len(sen) for sen in conll["sentences"][:sen_idx]])
        # make the output
        speakers = conll["speakers"][:sen_idx]
        sentences = conll["sentences"][:sen_idx]
        constituents = [
            con for con in conll["constituents"]
            if con[0] < num_words]
        ner = [
            con for con in conll["ner"]
            if con[0] < num_words]
        clusters = []
        for cluster in conll["clusters"]:
            cl = []
            for span in cluster:
                if span[0] < num_words:
                    cl.append(span)
            clusters.append(cl)
        ret_conll = {
            "speakers": speakers,
            "doc_key": conll["doc_key"],
            "sentences": sentences,
            "constituents": constituents,
            "clusters": clusters,
            "ner": ner
        }
        return CorefSample(ret_conll)

    @staticmethod
    def part_after_conll(c1, sen_idx):
        """ Only sentences [sen_idx:] will be kept, and all the structures of
            `clusters` are kept for convenience of concat.
        Args:
            c1: a CorefSample. the original sample
                conll: a conll-style dict. same infor to the original sample
            sen_idx: int. sentences with idx >= sen_idx will be preserved
        Returns:
            a CorefSample of a conll-part
        """
        conll = c1.dump()
        shift = sum([len(sen) for sen in conll["sentences"][:sen_idx]])
        # make the output
        speakers = conll["speakers"][sen_idx:]
        sentences = conll["sentences"][sen_idx:]
        constituents = [
            [con[0]-shift, con[1]-shift, con[2]]
            for con in conll["constituents"]
            if con[0] >= shift]
        ner = [
            [con[0]-shift, con[1]-shift, con[2]]
            for con in conll["ner"]
            if con[0] >= shift]
        clusters = [
            [
                [span[0]-shift, span[1]-shift]
                for span in cluster
                if span[0] >= shift]
            for cluster in conll["clusters"]]
        ret_conll = {
            "speakers": speakers,
            "doc_key": conll["doc_key"],
            "sentences": sentences,
            "constituents": constituents,
            "clusters": clusters,
            "ner": ner
        }
        return CorefSample(ret_conll)

    @staticmethod
    def concat_conll_parts(*args):
        """ merge many parts of a conll.
        Args:
            *args: many CorefSamples
                will soon be converted to conll_parts by `map dump`
            conll_parts: list of conll-style dict 
                Elements in conll_parts are assumed to be parts from 
                the same conll, generated by part_conll.
                The validity of the merge result is not checked in this func. 
                That is to say, the concat result may not be a valid conll, but
                still a conll-part, which should be postprocessed by 
                `remove_invalid_corefs_from_part` to form a valid CorefSample.

        """
        conll_parts = [c.dump() for c in args]
        assert len(conll_parts) > 0  # cannot do with empty list

        def recursively_process_list(f, ls):
            """ Apply `f` to every elem in `ls` (a nested list) recursively.
            Usages:
                recursively_process_list(lambda x: x+2, 1) = 3
                r_p_l(lambda x: x+2, [2, [3, 4]]) = [4, [5, 6]]
            """
            if isinstance(ls, list):
                return [recursively_process_list(f, elem) for elem in ls]
            else:
                return f(ls)
        num_words_c = [
            sum([len(sen) for sen in conll_part["sentences"]])
            for conll_part in conll_parts]
        # shift_c: counts the shift of c_idx-th conll
        def shift_c(c_idx): return sum(num_words_c[:c_idx])
        # shift_func: given shift, apply the shift recursively
        def shift_func_c(shift): return lambda ls: recursively_process_list(
            lambda x: x+shift if isinstance(x, int) else x, ls)
        # do the append
        c1 = conll_parts[0]
        speakers = c1["speakers"]
        sentences = c1["sentences"]
        constituents = c1["constituents"]
        clusters = c1["clusters"]
        ner = c1["ner"]

        for i in range(1, len(conll_parts)):
            conll_part = conll_parts[i]
            shift_func = shift_func_c(shift_c(i))
            speakers.extend(conll_part["speakers"])
            sentences.extend(conll_part["sentences"])
            constituents.extend(shift_func(conll_part["constituents"]))
            for j in range(len(clusters)):
                clusters[j].extend(shift_func(conll_part["clusters"])[j])
            ner.extend(shift_func(conll_part["ner"]))
        ret_conll = {
            "speakers": speakers,
            "doc_key": c1["doc_key"],
            "sentences": sentences,
            "constituents": constituents,
            "clusters": clusters,
            "ner": ner
        }
        return CorefSample(ret_conll)

    @staticmethod
    def remove_invalid_corefs_from_part(c1):
        """ Parts of CorefSamples/conlls may contain clusters that has only 
            0 or 1 span, which is not a valid one. 
            This function remove these invalid clusters from c1.clusters.
        Args:
            c1: CorefSample, a conll-part
        Returns:
            a CorefSample that passes check_data(tolerate_part=False)
        """
        clusters = c1.clusters.field_value
        c1.clusters = ListField(
            [cluster for cluster in clusters if len(cluster) > 1]
        )
        return c1

    # end of the definition of CorefSample


# some useful coref samples

def words(s): return s.split(" ")
def unwords(ws): return " ".join(ws)
# word("i love u .") == ["i", "love", "u", "."]
# unwords(["i", "love", "u", "."]) == "i love u ."

sens1 = [
    words("I love my pet Anna ."),
    words("She is my favorite .")
]
clusters1 = [[[2, 3], [4, 4], [6, 6]]]
conll1 = CorefSample.make_sample_from_sentences_and_clusters(
    sens1, clusters1, "1")

sens2 = [
    words("Bob 's wife Anna likes winter ."),
    words("However , he loves summer .")
]
clusters2 = [[[0, 2], [3, 3]], [[0, 0], [9, 9]]]
conll2 = CorefSample.make_sample_from_sentences_and_clusters(
    sens2, clusters2, "2")

sens3 = [words("Nothing .")]
clusters3 = []
conll3 = CorefSample.make_sample_from_sentences_and_clusters(
    sens3, clusters3, "3")

coref_sample1 = CorefSample(conll1)
coref_sample2 = CorefSample(conll2)
coref_sample3 = CorefSample(conll3)


if __name__ == "__main__":
    pprint(coref_sample1.dump())

    a1 = coref_sample1
    a2 = coref_sample2

    print("test: concat_two_conlls")
    a11 = CorefSample.concat_two_conlls(a1, a2)
    # pprint(a11.dump())

    print("test: part_conll")
    a21 = CorefSample.part_conll(CorefSample.concat_two_conlls(a1, a2), [0, 2])
    a22 = CorefSample.part_conll(CorefSample.concat_two_conlls(a1, a2), [0, 1])
    # pprint(a21.dump())
    # pprint(a22.dump())

    print("test: part_before_conll, part_after_conll")
    a31 = CorefSample.part_before_conll(a11, 2)
    a32 = CorefSample.part_after_conll(a11, 2)
    # pprint(a31.dump())
    # pprint(a32.dump())

    print("test: concat_conll_parts")
    a4 = CorefSample.concat_conll_parts(a31, a32)
    # pprint(a4.dump())
