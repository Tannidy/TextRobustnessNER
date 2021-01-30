"""
:type: function
:Size: 167MB
:Package Requirements: * **Java**

Model files for Stanford Parser.
`[page] <https://nlp.stanford.edu/software/lex-parser.shtml>`__
"""

from nltk.parse.stanford import StanfordDependencyParser
import os

from TextRobustness.common.utils import make_zip_downloader

NAME = "TProcess.StanfordDependencyParser"

URL = "https://nlp.fudan.edu.cn//TextRobustness/download/parser.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    return (
        __import__("nltk.parse.stanford")
        .parse.stanford.StanfordDependencyParser(
            path_to_jar=os.path.join(path, './stanford-parser.jar'),
            path_to_models_jar=os.path.join(path, "./stanford-parser-4.2.0-models.jar"),
            model_path=os.path.join(path, './englishPCFG.ser.gz'))
        .raw_parse
    )


if __name__ == "__main__":
    path = "/Users/wangxiao/code/python/RobustnessTool/TextRobustness/textrobustness/TextRobustness/common/res/TProcess.StanfordDependencyParser/"
    sent = "The brown fox quick jump over the lazy dog."
    dp = StanfordDependencyParser(path_to_jar=os.path.join(path, './stanford-parser.jar'),
            path_to_models_jar=os.path.join(path, "./stanford-parser-4.2.0-models.jar"),
            model_path=os.path.join(path, './englishPCFG.ser.gz'))
    parse, = dp.raw_parse(sent)
    print(parse.to_conll(4))
