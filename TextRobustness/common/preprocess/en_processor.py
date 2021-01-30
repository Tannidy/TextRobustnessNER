"""
TextProcessor Class
============================================

"""
import threading
from functools import reduce
import nltk

from TextRobustness.common.preprocess.res_load import ModelManager
from TextRobustness.common.preprocess.tokenizer import tokenize, untokenize


class TextProcessor:
    """ Text Processor class implement NER, POS tag, lexical tree parsing with ``nltk`` toolkit.

    TextProcessor is designed by single instance mode.

    """
    _instance_lock = threading.Lock()

    def __init__(self):
        self.nltk = __import__("nltk")
        self.__stanford_tokenizer = None
        self.__sent_tokenizer = None
        self.__lemmatize = None
        self.__delemmatize = None
        self.__ner = None
        self.__parser = None
        self.__dp_parser = None
        self.__wordnet = None

    # Single instance mode
    def __new__(cls, *args, **kwargs):
        if not hasattr(TextProcessor, "_instance"):
            with TextProcessor._instance_lock:
                if not hasattr(TextProcessor, "_instance"):
                    TextProcessor._instance = object.__new__(cls)
        return TextProcessor._instance

    @staticmethod
    def word_tokenize(sent):
        assert isinstance(sent, str)
        return tokenize(sent)

    @staticmethod
    def inverse_tokenize(tokens):
        assert isinstance(tokens, list)
        return untokenize(tokens)

    def sentence_tokenize(self, paras):
        assert isinstance(paras, str)
        if self.__sent_tokenizer is None:
            self.__sent_tokenizer = ModelManager.load("TProcess.NLTKSentTokenizer")

        return self.__sent_tokenizer(paras)

    def stanford_tokenize(self, sent):
        assert isinstance(sent, str)
        if self.__stanford_tokenizer is None:
            self.__stanford_tokenizer = ModelManager.load("TProcess.StanfordTokenizer")

        return self.__stanford_tokenizer(sent)

    @staticmethod
    def get_pos(sentence):
        """ POS tagging function.

        Example:
            TextProcessor().get_pos('All things in their being are good for something.')

            >> [('All', 'DT'),
                ('things', 'NNS'),
                ('in', 'IN'),
                ('their', 'PRP$'),
                ('being', 'VBG'),
                ('are', 'VBP'),
                ('good', 'JJ'),
                ('for', 'IN'),
                ('something', 'NN'),
                ('.', '.')]

        Args:
            sentence: str or list.
                A sentence which needs to be tokenized.

        Returns:
            Tokenized tokens with their POS tags.

        """
        assert isinstance(sentence, (str, list))
        tokens = tokenize(sentence) if isinstance(sentence, str) else sentence  # concatenate tokens

        return nltk.pos_tag(tokens)

    @staticmethod
    def index_word2char(tokens, start, end):
        """ Convert span idx from word level to char level.

        Args:
            tokens: list
                word seqs.
            start: int
                span start index of word level.
            end: int
                span end index of word level.

        Returns:
            char_start: int
            char_end: int

        """
        if max(start, end - 1) > len(tokens) - 1:
            raise ValueError("Word index {0} out of tokens list length {1}".format((start, end), len(tokens)))
        char_start = 0 if tokens[:start] is [] else reduce(lambda x, y: x + len(y) + 1, tokens[:start], 0)
        char_seq_len = 0 if tokens[start:end] is [] else reduce(lambda x, y: x + len(y) + 1, tokens[start:end], -1)

        return [char_start, char_start + char_seq_len]

    def get_ner(self, sentence, return_char_idx=True):
        """ NER function.

        This method uses NLTK tokenizer and Stanford NER toolkit which requires Java installed.

        Example:
            TextProcessor().get_ner('Lionel Messi is a football player from Argentina.')

            if return_word_index is False
            >>[('Lionel Messi', 0, 12, 'PERSON'),
               ('Argentina', 39, 48, 'LOCATION')]

            if return_word_index is True
            >>[('Lionel Messi', 0, 2, 'PERSON'),
               ('Argentina', 7, 8, 'LOCATION')]

        Args:
            sentence: str or list
                A sentence that we want to extract named entities.
            return_char_idx: bool
                if set True, return character start to end index, else return char start to end index.

        Returns:
            A list of tuples, *(entity, start, end, label)*

        """
        if self.__ner is None:
            self.__ner = ModelManager.load("TProcess.StanfordNER")

        if isinstance(sentence, list):
            tokens = sentence
        elif isinstance(sentence, str):
            tokens = self.word_tokenize(sentence)  # list of tokens
        else:
            raise ValueError('Support string or token list input, while your input type is {0}'.format(type(sentence)))

        ret = []
        ne_buffer = []
        ne_start_pos = 0
        ne_last_pos = 0
        ne_type = ""
        last_NE = False
        nes = self.__ner(tokens)

        for idx in range(len(tokens)):
            word, ne = nes[idx]

            if ne == "O":
                if last_NE:
                    last_NE = False
                    ret.append((" ".join(ne_buffer), ne_start_pos, ne_last_pos, ne_type))
            else:
                if (not last_NE) or (ne_type != ne):
                    if last_NE:
                        # append last ne
                        ret.append((" ".join(ne_buffer), ne_start_pos, ne_last_pos, ne_type))
                    # new entity
                    ne_start_pos = idx
                    ne_last_pos = idx + 1
                    ne_type = ne
                    ne_buffer = [word]
                    last_NE = True
                else:
                    ne_last_pos = idx + 1
                    ne_buffer.append(word)
        # handle the last NE
        if last_NE:
            ret.append((" ".join(ne_buffer), ne_start_pos, ne_last_pos, ne_type))

        # convert word index to char index
        if return_char_idx:
            char_ret = []
            for entity, start, end, ne_type in ret:
                char_ret.append(tuple([entity] + TextProcessor.index_word2char(tokens, start, end) + [ne_type]))
            ret = char_ret

        return ret

    def get_parser(self, sentence):
        """ Lexical tree parsing.

        This method uses Stanford LexParser to generate a lexical tree which requires Java installed.

        Example:
            TextProcessor().get_parser('Messi is a football player.')

            >>'(ROOT\n  (S\n    (NP (NNP Messi))\n    (VP (VBZ is) (NP (DT a) (NN football) (NN player)))\n    (. .)))'

        Args:
            sentence: str or list.
                A sentence needs to be parsed.

        Returns:
            The result tree of lexicalized parser in string format.

        """

        if self.__parser is None:
            self.__parser = ModelManager.load("TProcess.StanfordParser")
        assert isinstance(sentence, (str, list))
        sentence = untokenize(sentence) if isinstance(sentence, list) else sentence  # concatenate tokens

        return str(list(self.__parser(sentence))[0])

    def get_dep_parser(self, sentence):
        """ Dependency parsing.

        Example:
            TextProcessor().get_dep_parser('The quick brown fox jumps over the lazy dog.')

            >>
                The	DT	4	det
                quick	JJ	4	amod
                brown	JJ	4	amod
                fox	NN	5	nsubj
                jumps	VBZ	0	root
                over	IN	9	case
                the	DT	9	det
                lazy	JJ	9	amod
                dog	NN	5	obl

        Args:
            sentence: str or list.
                A sentence needs to be parsed.

        Returns:

        """
        if self.__dp_parser is None:
            self.__dp_parser = ModelManager.load("TProcess.StanfordDependencyParser")
        assert isinstance(sentence, (str, list))
        sentence = untokenize(sentence) if isinstance(sentence, list) else sentence  # concatenate tokens
        parse, = self.__dp_parser(sentence)

        return parse.to_conll(4)

    def get_lemmas(self, token_and_pos):
        """ Lemmatize function.

        This method uses ``nltk.WordNetLemmatier`` to lemmatize tokens.

        Args:
            token_and_pos: list,  *(token, POS)*.

        Returns:
            A lemma or a list of lemmas depends on your input.

        """
        if self.__lemmatize is None:
            self.__lemmatize = ModelManager.load("TProcess.NLTKWordNet").lemma

        if not isinstance(token_and_pos, list):
            return self.__lemmatize(token_and_pos[0], token_and_pos[1])
        else:
            return [self.__lemmatize(token, pos) for token, pos in token_and_pos]

    def get_all_lemmas(self, pos):
        """ Lemmatize function for all words in WordNet.

        This method uses ``nltk.WordNetLemmatier`` to lemmatize tokens.

        Args:
            pos: POS tag pr a list of POS tag.

        Returns:
            A list of lemmas that have the given pos tag.

        """
        if self.__lemmatize is None:
            self.__lemmatize = ModelManager.load("TProcess.NLTKWordNet").all_lemma

        if not isinstance(pos, list):
            return self.__lemmatize(pos)
        else:
            return [self.__lemmatize(_pos) for _pos in pos]

    def get_delemmas(self, lemma_and_pos):
        """ Delemmatize function.

        This method uses a pre-processed dict which maps (lemma, pos) to original token for delemmatizing.

        Args:
            lemma_and_pos: list or tuple.
                A tuple or a list of tuples, *(lemma, POS)*.

        Returns:
            A word or a list of words, each word represents the specific form of input lemma.

        """

        if self.__delemmatize is None:
            self.__delemmatize = ModelManager.load("TProcess.NLTKWordNetDelemma")
        if not isinstance(lemma_and_pos, list):
            token, pos = lemma_and_pos
            return (
                self.__delemmatize[token][pos]
                if (token in self.__delemmatize) and (pos in self.__delemmatize[token])
                else token
            )
        else:
            return [
                self.__delemmatize[token][pos]
                if (token in self.__delemmatize) and (pos in self.__delemmatize[token])
                else token
                for token, pos in lemma_and_pos
            ]

    def get_synsets(self, tokens_and_pos, lang="eng"):
        """ Get synsets from WordNet.

        This method uses NTLK WordNet to generate synsets, and uses "lesk" algorithm which
        is proposed by Michael E. Lesk in 1986, to screen the sense out.

        Args:
            tokens_and_pos: A list of tuples, *(token, POS)*.

        Returns:
            A list of str, represents the sense of each input token.
        """
        if self.__wordnet is None:
            self.__wordnet = ModelManager.load("TProcess.NLTKWordNet")

        if isinstance(tokens_and_pos, str):
            tokens_and_pos = self.get_pos(tokens_and_pos)

        def lesk(sentence, word, pos):
            synsets = self.__wordnet.synsets(word, lang=lang)

            if pos is not None:
                synsets = [ss for ss in synsets if str(ss.pos()) == pos]

            return synsets

        sentoken = []
        sentence = []
        # normalize pos tag
        for word, pos in tokens_and_pos:
            sentoken.append(word)
            sentence.append((word, self.normalize_pos(pos)))
        ret = []

        for word, pos in sentence:
            ret.append(lesk(sentoken, word, pos))
        return ret

    def get_antonyms(self, tokens_and_pos, lang="eng"):
        """ Get antonyms from WordNet.

        This method uses NTLK WordNet to generate antonyms, and uses "lesk" algorithm which
        is proposed by Michael E. Lesk in 1986, to screen the sense out.

        Args:
            tokens_and_pos: A list of tuples, *(token, POS)*.

        Returns:
            A list of str, represents the sense of each input token.
        """
        if self.__wordnet is None:
            self.__wordnet = ModelManager.load("TProcess.NLTKWordNet")

        if isinstance(tokens_and_pos, str):
            tokens_and_pos = self.get_pos(tokens_and_pos)

        def lesk(sentence, word, pos):
            synsets = self.__wordnet.synsets(word, lang=lang)
            antonyms = set()

            for synonym in synsets:
                for l in synonym.lemmas():
                    if l.antonyms():
                        antonyms.add(l.antonyms()[0].synset())

            if pos is not None:
                antonyms = [ss for ss in antonyms if str(ss.pos()) == pos]

            return antonyms

        sentoken = []
        sentence = []
        # normalize pos tag
        for word, pos in tokens_and_pos:
            sentoken.append(word)
            sentence.append((word, self.normalize_pos(pos)))
        ret = []

        for word, pos in sentence:
            ret.append(lesk(sentoken, word, pos))
        return ret

    def filter_candidates_by_pos(self, token_and_pos, candidates, lang="eng"):
        """ Filter synonyms not contain the same pos tag with given token.

        Args:
            token_and_pos: list/tuple
            candidates: list

        Returns:
            filtered candidates list.
        """
        def lesk(word, pos, candidates):
            can_word_pos = []

            for candidate in candidates:
                can_word, can_pos = nltk.tag.pos_tag([candidate])[0]
                can_word_pos.append([can_word, self.normalize_pos(can_pos)])

            if pos is not None:
                return [ss[0] for ss in can_word_pos if str(ss[1]) == pos]
            else:
                return []

        # normalize pos tag
        word, pos = token_and_pos

        return lesk(word, self.normalize_pos(pos), candidates)

    @staticmethod
    def normalize_pos(pos):
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

        return pp


text_processor = TextProcessor()


if __name__ == "__main__":
    x = 'Lionel Messi is a football player from Argentina.'
    processor = TextProcessor()
    print(processor.filter_candidates_by_pos(('good', 'NN'),
                                             ['well', 'great', 'geed', 'best', 'product', "commodity", "goods"]))
    print(processor, text_processor)
    print(text_processor.get_dep_parser(x))
    print(text_processor.get_ner(x))
    print(text_processor.get_ner(x, return_char_idx=False))
    print(text_processor.get_pos(x))
    print(text_processor.get_parser(x))

