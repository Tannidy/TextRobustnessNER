"""
:type: function
:Size: 11.7MB
:Package Requirements: * **Java**


Model files for Stanford pos tagger in nltk.
`[page] <https://nlp.stanford.edu/software/tagger.html>`__
"""
from TextRobustness.common.utils import make_zip_downloader
import os

NAME = "TProcess.StanfordTokenizer"

URL = "https://nlp.fudan.edu.cn//TextRobustness/download/tagger.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    return (
        __import__("nltk.tokenize.stanford")
        .tokenize.stanford.StanfordTokenizer(
            path_to_jar=os.path.join(path, "stanford-postagger-4.2.0.jar"),
        )
        .tokenize
    )
