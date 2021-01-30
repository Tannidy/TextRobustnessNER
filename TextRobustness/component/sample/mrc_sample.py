""" Manage text transformation for MRC.
    Heavily borrowed from adversarial-squad.
    For code in adversarial-squad, please check the following link:
    https://github.com/robinjia/adversarial-squad
"""

from TextRobustness.component.sample import Sample
from TextRobustness.component import Field
from TextRobustness.component.field import TextField
from TextRobustness.common.preprocess import text_processor

from nltk.corpus import wordnet as wn
from nltk.stem.lancaster import LancasterStemmer

__all__ = ['MRCSample']
STEMMER = LancasterStemmer()


class ConstituencyParse(object):
    """A CoreNLP constituency parse (or a node in a parse tree).

    Word-level constituents have |word| and |index| set and no children.
    Phrase-level constituents have no |word| or |index| and have at least one child.
    """

    def __init__(self, tag, children=None, word=None, index=None):
        self.tag = tag
        if children:
            self.children = children
        else:
            self.children = None
        self.word = word
        self.index = index

    @classmethod
    def _recursive_parse_corenlp(cls, s, j):
        tag = s.value
        childrens = []
        if s.child:
            for children in s.child:
                children, j = cls._recursive_parse_corenlp(children, j)
                childrens.append(children)
            return cls(tag, childrens), j
        else:
            return cls(tag, word=tag, index=j), j + 1

    @classmethod
    def from_corenlp(cls, s):
        tree, num_words = cls._recursive_parse_corenlp(s, 0)
        return tree

    def is_singleton(self):
        if self.word:
            return True
        if len(self.children) > 1:
            return False
        return self.children[0].is_singleton()

    def get_phrase(self):
        if self.word:
            return self.word
        toks = []
        for i, c in enumerate(self.children):
            p = c.get_phrase()
            if i == 0 or p.startswith("'"):
                toks.append(p)
            else:
                toks.append(' ' + p)
        return ''.join(toks)

    def get_start_index(self):
        if self.index is not None:
            return self.index
        return self.children[0].get_start_index()

    def get_end_index(self):
        if self.index is not None:
            return self.index + 1
        return self.children[-1].get_end_index()

    @classmethod
    def _recursive_replace_words(cls, tree, new_words, i):
        if tree.word:
            new_word = new_words[i]
            return cls(tree.tag, word=new_word, index=tree.index), i + 1
        new_children = []
        for c in tree.children:
            new_child, i = cls._recursive_replace_words(c, new_words, i)
            new_children.append(new_child)
        return cls(tree.tag, children=new_children), i

    @classmethod
    def replace_words(cls, tree, new_words):
        """Return a new tree, with new words replacing old ones."""
        new_tree, i = cls._recursive_replace_words(tree, new_words, 0)
        return new_tree


class MRCSample(Sample):
    def __init__(self, data, origin=None):
        super().__init__(data, origin=origin)

    def __repr__(self):
        return 'MRCSample'

    def check_data(self, data):
        assert 'context' in data and isinstance(data['context'], str)
        assert 'question' in data and isinstance(data['question'], str)
        assert 'answer' in data and isinstance(data['answer'], str)

    def load(self, data):
        """ Convert data dict which contains essential information to SASample.

        Args:
            data: dict
                contains 'x', 'y' keys.

        Returns:

        """
        self.context = TextField(data['context'])
        self.context._sentences = text_processor.sentence_tokenize(data['context'])
        self.question = TextField(data['question'])
        self.answer = Field(data['answer'])
        self.answer_start = Field(data['answer_start'])

    def dump(self):
        return {'context': self.context.text, 'question': self.question.text, 'answer': self.answer.field_value,
                'answer_start': self.answer_start.field_value}

    @classmethod
    def clone(cls, original_sample):
        """Deep copy self to a new sample.

        Args:
            original_sample: sample to be copied.
        Returns:
            Sample instance.

        """
        data = original_sample.dump()
        data['context'] = " ".join(original_sample.context.sentences)
        sample = cls(data)
        sample.origin = original_sample.origin

        return sample

    def feature_extract(self, sent):
        sent_pos = text_processor.get_pos(sent)
        sent_parser = text_processor.get_parser(sent)
        sent_lemma = text_processor.get_lemmas(sent_pos)
        ner = text_processor.get_ner(sent)
        tokens = []
        ner_num = len(ner)
        ner_idx = 0
        it = 0

        if ner_num > 0:
            _, ner_start, ner_end, ner_type = ner[ner_idx]
        for i, tok in enumerate(sent_pos):
            text, pos = tok
            it += sent[it:].find(text)

            if ner_num == 0:
                word_ner = 'O'
            else:
                if it > ner_end and ner_idx <= ner_num - 1:
                    ner_idx += 1
                    if ner_idx < ner_num:
                        _, ner_start, ner_end, ner_type = ner[ner_idx]

                if ner_idx == ner_num:
                    word_ner = "O"
                elif ner_start <= it < ner_end:
                    word_ner = ner_type
                else:
                    word_ner = "O"
            word = {'word': text,
                    'pos': pos,
                    'lemma': sent_lemma[i],
                    'ner': word_ner
                    }
            tokens.append(word)
        return tokens, sent_parser

    @staticmethod
    def run_conversion(question, answer, tokens, const_parse):
        """Convert the question and answer to a declarative sentence
        Args:
            question: question, str
            answer: answer, str
            tokens: the semantic tag dicts of question
            const_parse: the constituency parse of question

        Returns: a declarative sentence

        """

        for rule in CONVERSION_RULES:
            sent = rule.convert(question, answer, tokens, const_parse)
            if sent:
                return sent
        return None

    @staticmethod
    def convert_answer(answer, sent_tokens, question):
        """Replace the ground truth with fake answer based on specific rules
        Args:
            answer: ground truth, str
            sent_tokens: sentence dicts, like [{'word': 'Saint', 'pos': 'NNP', 'lemma': 'Saint', 'ner': 'PERSON'}...]
            question: question sentence, str

        Returns: fake answer, str

        """
        tokens = MRCSample.get_answer_tokens(sent_tokens, answer)
        determiner = MRCSample.get_determiner_for_answers(answer)
        for rule_name, func in ANSWER_RULES:
            new_ans = func(answer, tokens, question, determiner=determiner)
            if new_ans:
                return new_ans
        return None

    @staticmethod
    def alter_sentence(sample, nearby_word_dict=None, pos_tag_dict=None, rules=None):
        """
        Args:
            sample: sentence dicts, like [{'word': 'Saint', 'pos': 'NNP', 'lemma': 'Saint', 'ner': 'PERSON'}...]
            nearby_word_dict: the dictionary to search for nearby words
            pos_tag_dict: the dictionary to search for the most frequent pos tags
            rules: the rules to alter the sentence
            n: the number of altered sentences

        Returns: alter_sentence, alter_sentence dicts

        """
        used_words = [t['word'].lower() for t in sample]
        sentence = []
        new_sample = []
        for i, t in enumerate(sample):
            if t['word'].lower() in DO_NOT_ALTER:
                sentence.append(t['word'])
                new_sample.append(t)
                continue
            found = False
            for rule_name in rules:
                rule = rules[rule_name]
                new_words = rule(t, nearby_word_dict=nearby_word_dict,
                                 pos_tag_dict=pos_tag_dict)
                if new_words:
                    for nw in new_words:
                        if nw.lower() in used_words:
                            continue
                        if nw.lower() in BAD_ALTERATIONS:
                            continue
                        # Match capitalization
                        if t['word'] == t['word'].upper():
                            nw = nw.upper()
                        elif t['word'] == t['word'].title():
                            nw = nw.title()
                        found = True
                        sentence.append(nw)
                        new_sample.append({'word': nw,
                                           'lemma': nw,
                                           'pos': t['pos'],
                                           'ner': t['ner']
                                           })
                        break
                if found:
                    break
            if not found:
                sentence.append(t['word'])
                new_sample.append(t)

        return " ".join(sentence), new_sample

    @staticmethod
    def alter_special(token, **kwargs):
        """ Alter special tokens

        Args:
            token: the token to alter
            **kwargs:

        Returns:
            like 'US' ->  'UK'
        """
        w = token['word']
        if w in SPECIAL_ALTERATIONS:
            return [SPECIAL_ALTERATIONS[w]]
        return None

    @staticmethod
    def alter_wordnet_antonyms(token, **kwargs):
        """ Replace words with wordnet antonyms

        Args:
            token: the token to alter
            **kwargs:

        Returns:
            like good -> bad
        """
        if token['pos'] not in POS_TO_WORDNET:
            return None
        w = token['word'].lower()
        wn_pos = POS_TO_WORDNET[token['pos']]
        synsets = wn.synsets(w, wn_pos)
        if not synsets:
            return None
        synset = synsets[0]
        antonyms = []

        for lem in synset.lemmas():
            if lem.antonyms():
                for a in lem.antonyms():
                    new_word = a.name()
                    if '_' in a.name():
                        continue
                    antonyms.append(new_word)
        return antonyms

    @staticmethod
    def alter_wordnet_synonyms(token, **kwargs):
        """ Replace words with synonyms

        Args:
            token: the token to alter
            **kwargs:

        Returns:
            like 'good' -> 'great'
        """
        if token['pos'] not in POS_TO_WORDNET:
            return None
        w = token['word'].lower()
        wn_pos = POS_TO_WORDNET[token['pos']]
        synsets = wn.synsets(w, wn_pos)
        if not synsets:
            return None
        synonyms = []

        for syn in synsets:
            for syn_word in syn.lemma_names():
                if (
                        (syn_word != w)
                        and ("_" not in syn_word)
                ):
                    # WordNet can suggest phrases that are joined by '_' but we ignore phrases.
                    synonyms.append(syn_word)
        return synonyms

    @staticmethod
    def alter_nearby(pos_list, ignore_pos=False, is_ner=False):
        """ Alter words based on glove embedding space

        Args:
            pos_list: pos tags list
            ignore_pos: bool, whether to match pos tag
            is_ner: bool, indicate ner

        Returns:
            like 'Mary' -> 'Rose'
        """

        def func(token, nearby_word_dict=None, pos_tag_dict=None, **kwargs):
            if token['pos'] not in pos_list:
                return None
            if is_ner and token['ner'] not in ('PERSON', 'LOCATION', 'ORGANIZATION', 'MISC'):
                return None
            w = token['word'].lower()
            if w in 'war':
                return None
            if w not in nearby_word_dict:
                return None
            new_words = []
            w_stem = STEMMER.stem(w.replace('.', ''))

            for x in nearby_word_dict[w][1:]:
                new_word = x['word']
                # Make sure words aren't too similar (e.g. same stem)
                new_stem = STEMMER.stem(new_word.replace('.', ''))
                if w_stem.startswith(new_stem) or new_stem.startswith(w_stem):
                    continue
                if not ignore_pos:
                    # Check for POS tag match
                    if new_word not in pos_tag_dict:
                        continue
                    new_postag = pos_tag_dict[new_word]
                    if new_postag != token['pos']:
                        continue
                new_words.append(new_word)
            return new_words

        return func

    @staticmethod
    def alter_entity_type(token, **kwargs):
        """ Alter entity

        Args:
            token: the word to alter
            **kwargs:

        Returns:
            like 'London' -> 'Berlin'
        """
        pos = token['pos']
        ner = token['ner']
        word = token['word']
        is_abbrev = (word == word.upper() and not word == word.lower())
        if token['pos'] not in (
                'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS',
                'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'):
            # Don't alter non-content words
            return None
        if ner == 'PERSON':
            return ['Jackson']
        elif ner == 'LOCATION':
            return ['Berlin']
        elif ner == 'ORGANIZATION':
            if is_abbrev:
                return ['UNICEF']
            return ['Acme']
        elif ner == 'MISC':
            return ['Neptune']
        elif ner == 'NNP':
            if is_abbrev:
                return ['XKCD']
            return ['Dalek']
        elif pos == 'NNPS':
            return ['Daleks']

        return None

    @staticmethod
    def get_determiner_for_answers(a):
        words = a.split(' ')
        if words[0].lower() == 'the':
            return 'the'
        if words[0].lower() in ('a', 'an'):
            return 'a'
        return None

    @staticmethod
    def get_answer_tokens(sent_tokens, answer):
        """ Extract the pos, ner, lemma tags of answer tokens
        Args:
            sent_tokens: a list of dicts
            answer: answer, str

        Returns: a list of dicts
            like [
            {'word': 'Saint', 'pos': 'NNP', 'lemma': 'Saint', 'ner': 'PERSON'},
            {'word': 'Bernadette', 'pos': 'NNP', 'lemma': 'Bernadette', 'ner': 'PERSON'},
            {'word': 'Soubirous', 'pos': 'NNP', 'lemma': 'Soubirous', 'ner': 'PERSON'}]
            ]

        """
        sent = " ".join([t['word'] for t in sent_tokens])
        start = sent.find(answer)
        end = start + len(answer)
        tokens = []
        length = 0
        for i, tok in enumerate(sent_tokens):
            if length > end:
                break
            if start <= length < end:
                tokens.append(tok)
            length = length + 1 + len(tok['word'])
        return tokens

    @staticmethod
    def ans_entity_full(ner_tag, new_ans):
        """Returns a function that yields new_ans iff every token has |ner_tag|
        Args:
            ner_tag: ner tag, str
            new_ans:  answer dicts, like [{'word': 'Saint', 'pos': 'NNP', 'lemma': 'Saint', 'ner': 'PERSON'}...]

        Returns: fake answer, str
        """

        def func(a, tokens, q, **kwargs):
            for t in tokens:
                if t['ner'] != ner_tag:
                    return None
            return new_ans

        return func

    @staticmethod
    def ans_abbrev(new_ans):
        """
        Args:
            new_ans: answer words, str

        Returns: fake answer, str

        """
        def func(a, tokens, q, **kwargs):
            s = a
            if s == s.upper() and s != s.lower():
                return new_ans
            return None

        return func

    @staticmethod
    def ans_match_wh(wh_word, new_ans):
        """Returns a function that yields new_ans if the question starts with |wh_word|
        Args:
            wh_word: question word, str
            new_ans: answer dicts, like [{'word': 'Saint', 'pos': 'NNP', 'lemma': 'Saint', 'ner': 'PERSON'}...]

        Returns: fake answer, str

        """

        def func(a, tokens, q, **kwargs):
            if q.lower().startswith(wh_word + ' '):
                return new_ans
            return None

        return func

    @staticmethod
    def ans_pos(pos, new_ans, end=False, add_dt=False):
        """Returns a function that yields new_ans if the first/last token has |pos|
        Args:
            pos: pos tag, str
            new_ans: answer dicts, like [{'word': 'Saint', 'pos': 'NNP', 'lemma': 'Saint', 'ner': 'PERSON'}...]
            end: whether to use the last word to match the pos tag
            add_dt: whether to add a determiner, bool

        Returns: fake answer, str

        """

        def func(a, tokens, q, determiner, **kwargs):
            if end:
                t = tokens[-1]
            else:
                t = tokens[0]
            if t['pos'] != pos:
                return None
            if add_dt and determiner:
                return '%s %s' % (determiner, new_ans)
            return new_ans

        return func

    @staticmethod
    def ans_catch_all(new_ans):
        def func(a, tokens, q, **kwargs):
            return new_ans

        return func

    @staticmethod
    def compress_whnp(tree, inside_whnp=False):
        if not tree.children: return tree  # Reached leaf
        # Compress all children
        for i, c in enumerate(tree.children):
            tree.children[i] = MRCSample.compress_whnp(c, inside_whnp=inside_whnp or tree.tag == 'WHNP')

        if tree.tag != 'WHNP':
            if inside_whnp:
                # Wrap everything in an NP
                return ConstituencyParse('NP', children=[tree])
            return tree
        wh_word = None
        new_np_children = []
        new_siblings = []

        for i, c in enumerate(tree.children):
            if i == 0:
                if c.tag in ('WHNP', 'WHADJP', 'WHAVP', 'WHPP'):
                    wh_word = c.children[0]
                    new_np_children.extend(c.children[1:])
                elif c.tag in ('WDT', 'WP', 'WP$', 'WRB'):
                    wh_word = c
                else:
                    # No WH-word at start of WHNP
                    return tree
            else:
                if c.tag == 'SQ':  # Due to bad parse, SQ may show up here
                    new_siblings = tree.children[i:]
                    break
                # Wrap everything in an NP
                new_np_children.append(ConstituencyParse('NP', children=[c]))

        if new_np_children:
            new_np = ConstituencyParse('NP', children=new_np_children)
            new_tree = ConstituencyParse('WHNP', children=[wh_word, new_np])
        else:
            new_tree = tree
        if new_siblings:
            new_tree = ConstituencyParse('SBARQ', children=[new_tree] + new_siblings)

        return new_tree

    @staticmethod
    def read_const_parse(parse_str):
        """Construct a constituency tree based on constituency parser"""
        tree = ConstituencyParse.from_corenlp(parse_str)
        new_tree = MRCSample.compress_whnp(tree)
        return new_tree

    @staticmethod
    # Rules for converting questions into declarative sentences
    def fix_style(s):
        """Minor, general style fixes for questions."""
        s = s.replace('?', '')  # Delete question marks anywhere in sentence.
        s = s.strip(' .')
        if s[0] == s[0].lower():
            s = s[0].upper() + s[1:]
        return s + '.'

    def _check_match(node, pattern_tok):
        if pattern_tok in CONST_PARSE_MACROS:
            pattern_tok = CONST_PARSE_MACROS[pattern_tok]
        if ':' in pattern_tok:
            # ':' means you match the LHS category and start with something on the right
            lhs, rhs = pattern_tok.split(':')
            match_lhs = MRCSample._check_match(node, lhs)
            if not match_lhs:
                return False
            phrase = node.get_phrase().lower()
            retval = any(phrase.startswith(w) for w in rhs.split('/'))
            return retval
        elif '/' in pattern_tok:
            return any(MRCSample._check_match(node, t) for t in pattern_tok.split('/'))

        return ((pattern_tok.startswith('$') and pattern_tok[1:] == node.tag) or
                (node.word and pattern_tok.lower() == node.word.lower()))

    @staticmethod
    def _recursive_match_pattern(pattern_toks, stack, matches):
        """Recursively try to match a pattern, greedily."""
        if len(matches) == len(pattern_toks):
            # We matched everything in the pattern; also need stack to be empty
            return len(stack) == 0
        if len(stack) == 0:
            return False
        cur_tok = pattern_toks[len(matches)]
        node = stack.pop()
        # See if we match the current token at this level
        is_match = MRCSample._check_match(node, cur_tok)

        if is_match:
            cur_num_matches = len(matches)
            matches.append(node)
            new_stack = list(stack)
            success = MRCSample._recursive_match_pattern(pattern_toks, new_stack, matches)
            if success:
                return True
            # Backtrack
            while len(matches) > cur_num_matches:
                matches.pop()
        # Recurse to children
        if not node.children:
            return False  # No children to recurse on, we failed
        stack.extend(node.children[::-1])  # Leftmost children should be popped first

        return MRCSample._recursive_match_pattern(pattern_toks, stack, matches)

    @staticmethod
    def match_pattern(pattern, const_parse):
        pattern_toks = pattern.split(' ')
        whole_phrase = const_parse.get_phrase()
        if whole_phrase.endswith('?') or whole_phrase.endswith('.'):
            # Match trailing punctuation as needed
            pattern_toks.append(whole_phrase[-1])
        matches = []
        success = MRCSample._recursive_match_pattern(pattern_toks, [const_parse], matches)
        if success:
            return matches
        else:
            return None

    @staticmethod
    def convert_whp(node, q, a, tokens):
        if node.tag in ('WHNP', 'WHADJP', 'WHADVP', 'WHPP'):
            # Apply WHP rules
            cur_phrase = node.get_phrase()
            cur_tokens = tokens[node.get_start_index():node.get_end_index()]
            for i, r in enumerate(WHP_RULES):
                phrase = r.convert(cur_phrase, a, cur_tokens, node, run_fix_style=False)
                if phrase:
                    return phrase
        return None


class ConversionRule(object):
    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        raise NotImplementedError


class ConstituencyRule(ConversionRule):
    """A rule for converting question to sentence based on constituency parse."""

    def __init__(self, in_pattern, out_pattern, postproc=None):
        self.in_pattern = in_pattern  # e.g. "where did $NP $VP"
        self.out_pattern = out_pattern
        # e.g. "{1} did {2} at {0}."  Answer is always 0
        self.name = in_pattern
        if postproc:
            self.postproc = postproc
        else:
            self.postproc = {}

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        pattern_toks = self.in_pattern.split(' ')  # Don't care about trailing punctuation
        match = MRCSample.match_pattern(self.in_pattern, const_parse)
        appended_clause = False

        if not match:
            # Try adding a PP at the beginning
            appended_clause = True
            new_pattern = '$PP , ' + self.in_pattern
            pattern_toks = new_pattern.split(' ')
            match = MRCSample.match_pattern(new_pattern, const_parse)
        if not match:
            # Try adding an SBAR at the beginning
            new_pattern = '$SBAR , ' + self.in_pattern
            pattern_toks = new_pattern.split(' ')
            match = MRCSample.match_pattern(new_pattern, const_parse)
        if not match:
            return None
        appended_clause_match = None
        fmt_args = [a]

        for t, m in zip(pattern_toks, match):
            if t.startswith('$') or '/' in t:
                # First check if it's a WHP
                phrase = MRCSample.convert_whp(m, q, a, tokens)
                if not phrase:
                    phrase = m.get_phrase()
                fmt_args.append(phrase)
        if appended_clause:
            appended_clause_match = fmt_args[1]
            fmt_args = [a] + fmt_args[2:]
        output = self.gen_output(fmt_args)
        if appended_clause:
            output = appended_clause_match + ', ' + output
        if run_fix_style:
            output = MRCSample.fix_style(output)

        return output

    def gen_output(self, fmt_args):
        """By default, use self.out_pattern.  Can be overridden."""
        return self.out_pattern.format(*fmt_args)


class ReplaceRule(ConversionRule):
    """A simple rule that replaces some tokens with the answer."""

    def __init__(self, target, replacement='{}', start=False):
        self.target = target
        self.replacement = replacement
        self.name = 'replace(%s)' % target
        self.start = start

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        t_toks = self.target.split(' ')
        q_toks = q.rstrip('?.').split(' ')
        replacement_text = self.replacement.format(a)

        for i in range(len(q_toks)):
            if self.start and i != 0:
                continue
            if ' '.join(q_toks[i:i + len(t_toks)]).rstrip(',').lower() == self.target:
                begin = q_toks[:i]
                end = q_toks[i + len(t_toks):]
                output = ' '.join(begin + [replacement_text] + end)
                if run_fix_style:
                    output = MRCSample.fix_style(output)
                return output
        return None


class FindWHPRule(ConversionRule):
    """A rule that looks for $WHP's from right to left and does replacements."""
    name = 'FindWHP'

    def _recursive_convert(self, node, q, a, tokens, found_whp):
        if node.word: return node.word, found_whp
        if not found_whp:
            whp_phrase = MRCSample.convert_whp(node, q, a, tokens)
            if whp_phrase:
                return whp_phrase, True
        child_phrases = []

        for c in node.children[::-1]:
            c_phrase, found_whp = self._recursive_convert(c, q, a, tokens, found_whp)
            child_phrases.append(c_phrase)
        out_toks = []

        for i, p in enumerate(child_phrases[::-1]):
            if i == 0 or p.startswith("'"):
                out_toks.append(p)
            else:
                out_toks.append(' ' + p)

        return ''.join(out_toks), found_whp

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        out_phrase, found_whp = self._recursive_convert(const_parse, q, a, tokens, False)
        if found_whp:
            if run_fix_style:
                out_phrase = MRCSample.fix_style(out_phrase)
            return out_phrase
        return None


class AnswerRule(ConversionRule):
    """Just return the answer."""
    name = 'AnswerRule'

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        return a


POS_TO_WORDNET = {
    'NN': wn.NOUN,
    'JJ': wn.ADJ,
    'JJR': wn.ADJ,
    'JJS': wn.ADJ,
}
CONST_PARSE_MACROS = {
    '$Noun': '$NP/$NN/$NNS/$NNP/$NNPS',
    '$Verb': '$VB/$VBD/$VBP/$VBZ',
    '$Part': '$VBN/$VG',
    '$Be': 'is/are/was/were',
    '$Do': "do/did/does/don't/didn't/doesn't",
    '$WHP': '$WHADJP/$WHADVP/$WHNP/$WHPP',
}
SPECIAL_ALTERATIONS = {
    'States': 'Kingdom',
    'US': 'UK',
    'U.S': 'U.K.',
    'U.S.': 'U.K.',
    'UK': 'US',
    'U.K.': 'U.S.',
    'U.K': 'U.S.',
    'largest': 'smallest',
    'smallest': 'largest',
    'highest': 'lowest',
    'lowest': 'highest',
    'May': 'April',
    'Peyton': 'Trevor',
}

DO_NOT_ALTER = ['many', 'such', 'few', 'much', 'other', 'same', 'general',
                'type', 'record', 'kind', 'sort', 'part', 'form', 'terms', 'use',
                'place', 'way', 'old', 'young', 'bowl', 'united', 'one', 'ans_mask'
                'likely', 'different', 'square', 'war', 'republic', 'doctor', 'color']
BAD_ALTERATIONS = ['mx2004', 'planet', 'u.s.', 'Http://Www.Co.Mo.Md.Us']

MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
          'august', 'september', 'october', 'november', 'december']
ANSWER_RULES = [
    ('ner_person', MRCSample.ans_entity_full('PERSON', 'Jeff Dean')),
    ('ner_location', MRCSample.ans_entity_full('LOCATION', 'Chicago')),
    ('ner_organization', MRCSample.ans_entity_full('ORGANIZATION', 'Stark Industries')),
    ('ner_misc', MRCSample.ans_entity_full('MISC', 'Jupiter')),
    ('abbrev', MRCSample.ans_abbrev('LSTM')),
    ('wh_who', MRCSample.ans_match_wh('who', 'Jeff Dean')),
    ('wh_when', MRCSample.ans_match_wh('when', '1956')),
    ('wh_where', MRCSample.ans_match_wh('where', 'Chicago')),
    ('wh_where', MRCSample.ans_match_wh('how many', '42')),
    # Starts with verb
    ('pos_begin_vb', MRCSample.ans_pos('VB', 'learn')),
    ('pos_end_vbd', MRCSample.ans_pos('VBD', 'learned')),
    ('pos_end_vbg', MRCSample.ans_pos('VBG', 'learning')),
    ('pos_end_vbp', MRCSample.ans_pos('VBP', 'learns')),
    ('pos_end_vbz', MRCSample.ans_pos('VBZ', 'learns')),
    # Ends with some POS tag
    ('pos_end_nn', MRCSample.ans_pos('NN', 'hamster', end=True, add_dt=True)),
    ('pos_end_nnp', MRCSample.ans_pos('NNP', 'Central Park', end=True, add_dt=True)),
    ('pos_end_nns', MRCSample.ans_pos('NNS', 'hamsters', end=True, add_dt=True)),
    ('pos_end_nnps', MRCSample.ans_pos('NNPS', 'Kew Gardens', end=True, add_dt=True)),
    ('pos_end_jj', MRCSample.ans_pos('JJ', 'deep', end=True)),
    ('pos_end_jjr', MRCSample.ans_pos('JJR', 'deeper', end=True)),
    ('pos_end_jjs', MRCSample.ans_pos('JJS', 'deepest', end=True)),
    ('pos_end_rb', MRCSample.ans_pos('RB', 'silently', end=True)),
    ('pos_end_vbg', MRCSample.ans_pos('VBG', 'learning', end=True)),
    ('catch_all', MRCSample.ans_catch_all('aliens')),
]

MOD_ANSWER_RULES = [
    ('ner_person', MRCSample.ans_entity_full('PERSON', 'Charles Babbage')),
    ('ner_location', MRCSample.ans_entity_full('LOCATION', 'Stockholm')),
    ('ner_organization', MRCSample.ans_entity_full('ORGANIZATION', 'Acme Corporation')),
    ('ner_misc', MRCSample.ans_entity_full('MISC', 'Soylent')),
    ('abbrev', MRCSample.ans_abbrev('PCFG')),
    ('wh_who', MRCSample.ans_match_wh('who', 'Charles Babbage')),
    ('wh_when', MRCSample.ans_match_wh('when', '2004')),
    ('wh_where', MRCSample.ans_match_wh('where', 'Stockholm')),
    ('wh_where', MRCSample.ans_match_wh('how many', '200')),
    # Starts with verb
    ('pos_begin_vb', MRCSample.ans_pos('VB', 'run')),
    ('pos_end_vbd', MRCSample.ans_pos('VBD', 'ran')),
    ('pos_end_vbg', MRCSample.ans_pos('VBG', 'running')),
    ('pos_end_vbp', MRCSample.ans_pos('VBP', 'runs')),
    ('pos_end_vbz', MRCSample.ans_pos('VBZ', 'runs')),
    # Ends with some POS tag
    ('pos_end_nn', MRCSample.ans_pos('NN', 'apple', end=True, add_dt=True)),
    ('pos_end_nnp', MRCSample.ans_pos('NNP', 'Sears Tower', end=True, add_dt=True)),
    ('pos_end_nns', MRCSample.ans_pos('NNS', 'apples', end=True, add_dt=True)),
    ('pos_end_nnps', MRCSample.ans_pos('NNPS', 'Hobbits', end=True, add_dt=True)),
    ('pos_end_jj', MRCSample.ans_pos('JJ', 'blue', end=True)),
    ('pos_end_jjr', MRCSample.ans_pos('JJR', 'bluer', end=True)),
    ('pos_end_jjs', MRCSample.ans_pos('JJS', 'bluest', end=True)),
    ('pos_end_rb', MRCSample.ans_pos('RB', 'quickly', end=True)),
    ('pos_end_vbg', MRCSample.ans_pos('VBG', 'running', end=True)),
    ('catch_all', MRCSample.ans_catch_all('cosmic rays')),
]


CONVERSION_RULES = [
    # Special rules
    ConstituencyRule('$WHP:what $Be $NP called that $VP', '{2} that {3} {1} called {1}'),

    # What type of X
    ConstituencyRule("$WHP:what/which type/genre/kind/group of $NP/$Noun $Be $NP", '{5} {4} a {1} {3}'),
    ConstituencyRule("$WHP:what/which type/genre/kind/group of $NP/$Noun $Be $VP", '{1} {3} {4} {5}'),
    ConstituencyRule("$WHP:what/which type/genre/kind/group of $NP $VP", '{1} {3} {4}'),

    # How $JJ
    ConstituencyRule('how $JJ $Be $NP $IN $NP', '{3} {2} {0} {1} {4} {5}'),
    ConstituencyRule('how $JJ $Be $NP $SBAR', '{3} {2} {0} {1} {4}'),
    ConstituencyRule('how $JJ $Be $NP', '{3} {2} {0} {1}'),

    # When/where $Verb
    ConstituencyRule('$WHP:when/where $Do $NP', '{3} occurred in {1}'),
    ConstituencyRule('$WHP:when/where $Do $NP $Verb', '{3} {4} in {1}', {4: 'tense-2'}),
    ConstituencyRule('$WHP:when/where $Do $NP $Verb $NP/$PP', '{3} {4} {5} in {1}', {4: 'tense-2'}),
    ConstituencyRule('$WHP:when/where $Do $NP $Verb $NP $PP', '{3} {4} {5} {6} in {1}', {4: 'tense-2'}),
    ConstituencyRule('$WHP:when/where $Be $NP', '{3} {2} in {1}'),
    ConstituencyRule('$WHP:when/where $Verb $NP $VP/$ADJP', '{3} {2} {4} in {1}'),

    # What/who/how $Do
    ConstituencyRule("$WHP:what/which/who $Do $NP do", '{3} {1}', {0: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb", '{3} {4} {1}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $IN/$NP", '{3} {4} {5} {1}', {4: 'tense-2', 0: 'vbg'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $PP", '{3} {4} {1} {5}', {4: 'tense-2', 0: 'vbg'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $NP $VP", '{3} {4} {5} {6} {1}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb to $VB", '{3} {4} to {5} {1}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb to $VB $VP", '{3} {4} to {5} {1} {6}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $NP $IN $VP", '{3} {4} {5} {6} {1} {7}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $PP/$S/$VP/$SBAR/$SQ", '{3} {4} {1} {5}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $PP $PP/$S/$VP/$SBAR", '{3} {4} {1} {5} {6}',
                     {4: 'tense-2'}),

    # What/who/how $Be
    # Watch out for things that end in a preposition
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP of $NP $Verb/$Part $IN", '{3} of {4} {2} {5} {6} {1}'),
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP $NP $IN", '{3} {2} {4} {5} {1}'),
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP $VP/$IN", '{3} {2} {4} {1}'),
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP $IN $NP/$VP", '{1} {2} {3} {4} {5}'),
    ConstituencyRule('$WHP:what/which/who $Be/$MD $NP $Verb $PP', '{3} {2} {4} {1} {5}'),
    ConstituencyRule('$WHP:what/which/who $Be/$MD $NP/$VP/$PP', '{1} {2} {3}'),
    ConstituencyRule("$WHP:how $Be/$MD $NP $VP", '{3} {2} {4} by {1}'),

    # What/who $Verb
    ConstituencyRule("$WHP:what/which/who $VP", '{1} {2}'),

    # $IN what/which $NP
    ConstituencyRule('$IN what/which $NP $Do $NP $Verb $NP', '{5} {6} {7} {1} the {3} of {0}',
                     {1: 'lower', 6: 'tense-4'}),
    ConstituencyRule('$IN what/which $NP $Be $NP $VP/$ADJP', '{5} {4} {6} {1} the {3} of {0}',
                     {1: 'lower'}),
    ConstituencyRule('$IN what/which $NP $Verb $NP/$ADJP $VP', '{5} {4} {6} {1} the {3} of {0}',
                     {1: 'lower'}),
    FindWHPRule(),
]
WHP_RULES = [
    # WHPP rules
    ConstituencyRule('$IN what/which type/sort/kind/group of $NP/$Noun', '{1} {0} {4}'),
    ConstituencyRule('$IN what/which type/sort/kind/group of $NP/$Noun $PP', '{1} {0} {4} {5}'),
    ConstituencyRule('$IN what/which $NP', '{1} the {3} of {0}'),
    ConstituencyRule('$IN $WP/$WDT', '{1} {0}'),

    # what/which
    ConstituencyRule('what/which type/sort/kind/group of $NP/$Noun', '{0} {3}'),
    ConstituencyRule('what/which type/sort/kind/group of $NP/$Noun $PP', '{0} {3} {4}'),
    ConstituencyRule('what/which $NP', 'the {2} of {0}'),

    # How many
    ConstituencyRule('how many/much $NP', '{0} {2}'),

    # Replace
    ReplaceRule('what'),
    ReplaceRule('who'),
    ReplaceRule('how many'),
    ReplaceRule('how much'),
    ReplaceRule('which'),
    ReplaceRule('where'),
    ReplaceRule('when'),
    ReplaceRule('why'),
    ReplaceRule('how'),

    # Just give the answer
    AnswerRule(),
]