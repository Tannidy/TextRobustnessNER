"""
:type: nltk.WordNetCorpusReader
:Size: 10.283MB

Model files for wordnet in nltk.
`[page] <http://wordnet.princeton.edu/>`__
"""
from TextRobustness.common.utils import make_zip_downloader

NAME = "TProcess.NLTKWordNet"

URL = "https://nlp.fudan.edu.cn//TextRobustness/download/wordnet.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    wnc = __import__("nltk").corpus.WordNetCorpusReader(path, None)

    def lemma(word, pos):
        pp = "n"
        if pos in ["a", "r", "n", "v", "s"]:
            pp = pos
        else:
            if pos[:2] == "JJ":
                pp = "a"
            elif pos[:2] == "VB":
                pp = "v"
            elif pos[:2] == "NN":
                pp = "n"
            elif pos[:2] == "RB":
                pp = "r"
            else:
                pp = None
        if pp is None:  # do not need lemmatization
            return word
        lemmas = wnc._morphy(word, pp)
        return min(lemmas, key=len) if len(lemmas) > 0 else word

    def all_lemma(pos):
        return wnc.all_lemma_names(pos)

    wnc.lemma = lemma
    wnc.all_lemma = all_lemma
    return wnc
