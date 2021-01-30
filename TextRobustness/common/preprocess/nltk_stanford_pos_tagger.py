"""
:type: function
:Size: 11.7MB
:Package Requirements: * **Java**


Model files for pos tagger in nltk.
`[page] <https://nlp.stanford.edu/software/tagger.html>`__
"""
from TextRobustness.common.utils import make_zip_downloader
import os

NAME = "TProcess.NLTKStanfordPosTagger"
URL = "https://nlp.fudan.edu.cn//TextRobustness/download/tagger.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    ret = __import__("nltk").tag.\
        StanfordPOSTagger(model_filename=os.path.join(path, 'english-bidirectional-distsim.tagger'),
                          path_to_jar=os.path.join(path, 'stanford-postagger-4.2.0.jar'))
    return ret.tag
