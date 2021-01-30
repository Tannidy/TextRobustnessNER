import re
import random
from abc import ABC
from copy import deepcopy
from nltk.corpus import wordnet as wn
from TextRobustness.transformation import Transformation
from TextRobustness.common.settings import NEGATIVE_WORDS_LIST, DEGREE_WORD_LIST, PHRASE_LIST


class ABSATransformation(Transformation, ABC):
    """ An class that supply methods for ABSA task data transformation.
    """

    def __init__(self):
        super().__init__()

        self.negative_words_list = sorted(NEGATIVE_WORDS_LIST,
                                          key=lambda s: len(s), reverse=True)
        self.tokenize = self.processor.word_tokenize

    @staticmethod
    def untokenize(words):
        """ Untokenizing a text undoes the tokenizing operation, restoring
            punctuation and spaces to the places that people expect them to be.
            Ideally, `untokenize(tokenize(text))` should be identical to `text`,
            except for line breaks.

            Args:
                words:

            Returns:

        """
        text = ' '.join(words)
        step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',
                                                                     '...')
        step2 = step1.replace(" ( ", " (").replace(" ) ", ") ").replace(' - ',
                                                                        '-').replace(
            ' / ', '/')
        step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
        step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
        step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
            "can not", "cannot")
        step6 = step5.replace(" ` ", " '")
        step7 = step6.replace("DELETE", "")
        step8 = re.sub(r"\s{2,}", " ", step7)
        return step8.strip()

    def tokenize_term_list(self, sample):
        """Tokenize the term list of ABSASample.

        Args:
            sample: dict
        Returns:
            term_to_position_list: list
        """
        term_to_position_list = {}
        copy_sent = sample.x
        term_list = sample.term_list

        for term_id in term_list:
            if term_id not in term_to_position_list:
                term_to_position_list[term_id] = {}
            opinion_to_position_list = []
            opinions = term_list[term_id]['opinion_words']
            opinions_spans = term_list[term_id]['opinion_position']
            polarity = term_list[term_id]['polarity']
            for i in range(len(opinions)):
                position = opinions_spans[i]
                opinion_from = position[0]
                opinion_to = position[1]
                left = self.tokenize(copy_sent[:opinion_from].strip())
                opinion = self.tokenize(copy_sent[opinion_from:opinion_to].strip())
                opinion_to_position_list.append(
                    [' '.join(opinion), [len(left), len(left) + len(opinion)]])

            term_from = term_list[term_id]['from']
            term_to = term_list[term_id]['to']
            left = self.tokenize(copy_sent[:term_from].strip())
            aspect = self.tokenize(copy_sent[term_from:term_to].strip())
            term_to_position_list[term_id]['id'] = term_id
            term_to_position_list[term_id]['term'] = term_list[term_id]['term']
            term_to_position_list[term_id]['from'] = len(left)
            term_to_position_list[term_id]['to'] = len(left) + len(aspect)
            term_to_position_list[term_id]['polarity'] = polarity
            term_to_position_list[term_id]['opinions'] = opinion_to_position_list

        return term_to_position_list

    def reverse(self, words_list, opinions):
        """Reverse the polarity of opinions.

        Args:
            words_list: list
            opinions: list
        Returns:
            new_words: list
            new_opi_words: list
        """
        trans_words = deepcopy(words_list)
        trans_opinion_words = []
        from_to = []

        for i in range(len(opinions)):
            opinion_position = opinions[i][1]
            opinion_from = opinion_position[0]
            opinion_to = opinion_position[1]
            trans_words, opinion_from, opinion_to, has_neg = self.check_negation(
                trans_words, opinion_from, opinion_to)
            opinion_list = trans_words[opinion_from:opinion_to]
            opinion_words = trans_words[opinion_from:opinion_to]
            if len(opinion_list) == 1:
                trans_opinion_words = self.reverse_opinion(trans_words, trans_opinion_words,
                                                           from_to, opinion_from, opinion_to, has_neg)
            elif len(opinion_list) > 1:
                if has_neg:
                    trans_opinion_words.append(
                        [opinion_from, opinion_to, self.untokenize(opinion_words)])
                else:
                    # negate the closest verb
                    trans_opinion_words.append(
                        [opinion_from, opinion_to, self.untokenize(
                            ['not ' + opinion_words[0]] + opinion_words[1:])])
        for nopi in trans_opinion_words:
            trans_words[nopi[0]:nopi[1]] = [nopi[2]]

        return trans_words, trans_opinion_words

    def exaggerate(self, words_list, opinions):
        """Exaggerate the opinion words.

        Args:
            words_list: list
            opinions: list
        Returns:
            new_words: list
            new_opi_words: list
        """
        new_words = deepcopy(words_list)
        new_opi_words = []
        for i in range(len(opinions)):
            opi_position = opinions[i][1]
            opi_from = opi_position[0]
            opi_to = opi_position[1]

            new_words = self.add_degree_words(new_words, opi_from, opi_to)
            new_opi_word = self.untokenize(new_words[opi_from:opi_to])
            new_opi_words.append([opi_from, opi_to, new_opi_word])

        return new_words, new_opi_words

    def get_postag(self, sentence, start, end):
        """Get the postag.

        Args:
            sentence: str
            start: int
            end: int
        Returns:
            tags: list
        """

        tags = self.processor.get_pos(sentence)
        if end != -1:
            return tags[start:end]
        else:
            return tags[start:]

    @staticmethod
    def get_antonym_words(word):
        """Get antonym words.

        Args:
            word: str
        Returns:
            antonyms: str
        """
        antonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.add(lemma.antonyms()[0].name())
        return antonyms

    def refine_candidate(self, words_list, opi_from, opi_to, candidate_list):
        """Refine the candidate opinion words.

        Args:
            words_list: list
            opi_from: int
            opi_to: int
            candidate_list: list
        Returns:
            antonyms: list
        """
        if len(words_list) == 0:
            return []
        postag_list = self.get_postag(words_list, 0, -1)
        postag_list = [t[1] for t in postag_list]
        refined_candi = self.get_candidate(candidate_list, words_list, postag_list, opi_from, opi_to)
        return refined_candi

    @staticmethod
    def get_word2id(text, lower=True):
        """Get the index of words in sentence.

        Args:
            text: str
            lower: bool
        Returns:
            word2idx: dict
        """
        word2idx = {}
        idx = 1
        if lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1
        return word2idx

    @staticmethod
    def add_degree_words(word_list, from_idx, to_idx):
        """Add the degree words to sentence.

        Args:
            word_list: list
            from_idx: int
            to_idx: int
        Returns:
            new_words: list
        """

        candidate_list = DEGREE_WORD_LIST
        select = random.randint(0, len(candidate_list) - 1)
        opi = [' '.join([candidate_list[select]] + word_list[from_idx:to_idx])]
        new_words = word_list[:from_idx] + opi + word_list[to_idx:]
        return new_words

    @staticmethod
    def get_phrase(word, opi, ptree):
        """Get phrase.

        Args:
            word: list
            opi: list
            ptree: list
        Returns:
            phrase: list
        """

        phrase = []
        for node in ptree.subtrees(filter=lambda t: t.label() in PHRASE_LIST):
            if node.label() == 'NP':
                if node.right_sibling() is not None and node.right_sibling().label() == 'VP':
                    continue
            if node.label() == 'VP':
                if node.left_sibling() is not None and node.left_sibling().label() == 'NP':
                    continue
            if ''.join(word.split(' ')) in ''.join(node.leaves()) and ''.join(
                    opi.split(' ')) in ''.join(node.leaves()):
                phrase.append(node.leaves())
        phrase = sorted(phrase, key=len, reverse=True)

        return phrase

    @staticmethod
    def get_conjunction_idx(trans_words, aspect_term, conjunction_list):
        """Get the index of conjunction words in  conjunction_list.

        Args:
            trans_words: list
            aspect_term: list
            conjunction_list: list
        Returns:
            trans_idx: list
        """
        conjunction_idx = []
        trans_idx = None
        term = aspect_term['term']
        term_from = aspect_term['from']
        term_to = aspect_term['to']
        distance_to_term = len(trans_words)
        for idx, word in enumerate(trans_words):
            if word.lower() in conjunction_list and word.lower() not in term.lower():
                conjunction_idx.append(idx)
        for idx in conjunction_idx:
            if idx > term_to and idx - term_to < distance_to_term:
                distance_to_term = idx - term_to
                trans_idx = idx
            if idx < term_from and term_to - idx:
                distance_to_term = term_to - idx
                trans_idx = idx
        return trans_idx

    def get_sentence(self, trans_words, sentence):
        """Untokenize and uppercase to get an output sentence.

        Args:
            trans_words: list
            sentence: list
        Returns:
            trans_sentence: list
        """
        trans_sentence = self.untokenize(trans_words)
        if sentence[0].isupper():
            trans_sentence = trans_sentence[0].upper() + trans_sentence[1:]
        return trans_sentence

    def get_term_span(self, trans_sentence, term):
        """Get the span of term in trans_sentence.

        Args:
            trans_sentence: list
            term: list
        Returns:
            span_from: int
            span_to: int
        """
        span_from = 0
        char_from = 0
        char_sentence = ''.join(self.tokenize(trans_sentence))
        char_term = ''.join(self.tokenize(term))
        for idx in range(len(char_sentence)):
            if char_sentence[idx:idx + len(char_term)] == char_term:
                char_from = len(char_sentence[:idx])
                break
        trans_from = 0
        for idx in range(len(trans_sentence)):
            if trans_sentence[idx] != ' ':
                trans_from += 1
            if trans_from == char_from and char_from != 0 and trans_sentence[idx + 1] != ' ':
                span_from = idx + 1
                break
            if trans_from == char_from and char_from != 0 and trans_sentence[idx + 1] == ' ':
                span_from = idx + 2
                break
        span_to = span_from + len(term)
        return span_from, span_to

    def get_candidate(self, candidate_list, words_list, postag_list, opi_from, opi_to):
        """Get the candidate opinion words from words_list.

        Args:
            candidate_list: list
            words_list: list
            postag_list: list
            opi_from: int
            opi_to: int
        Returns:
            refined_candi: list
        """
        refined_candi = []
        for candidate in candidate_list:
            opi = words_list[opi_from:opi_to][0]
            isupper = opi[0].isupper()
            allupper = opi.isupper()
            if allupper:
                candidate = candidate.upper()
            elif isupper:
                candidate = candidate[0].upper() + candidate[1:]
            if opi_from == 0:
                candidate = candidate[0].upper() + candidate[1:]

            new_words = words_list[:opi_from] + [candidate] + words_list[opi_to:]
            # check pos tag
            new_postag_list = self.get_postag(new_words, 0, -1)
            new_postag_list = [t[1] for t in new_postag_list]

            if len([i for i, j in zip(postag_list[opi_from:opi_to],
                                      new_postag_list[opi_from:opi_to]) if
                    i != j]) != 0:
                continue
            refined_candi.append(candidate)
        return refined_candi

    def check_negation(self, trans_words, opinion_from, opinion_to):
        """Check the negation words in trans_words and delete them.

        Args:
            trans_words: list
            opinion_from: int
            opinion_to: int
        Returns:
            trans_words: list
            opinion_from: int
            opinion_to: int
            has_neg: bool
        """
        has_neg = False
        for w in self.negative_words_list:
            ws = self.tokenize(w)
            for j in range(opinion_from, opinion_to - len(ws) + 1):
                trans_words_ = ' '.join(trans_words[j:j + len(ws)])
                ws_ = ' '.join(ws)
                if trans_words_.lower() == ws_.lower():
                    if j > opinion_from:
                        opinion_to = opinion_to - len(ws)
                        trans_words[j: j + len(ws)] = ['DELETE'] * len(ws)
                        has_neg = True
                        break
                    else:
                        opinion_from = j + len(ws)
                        trans_words[j: j + len(ws)] = ['DELETE'] * len(ws)
                        has_neg = True
                        break
            if has_neg:
                break
        return trans_words, opinion_from, opinion_to, has_neg

    def reverse_opinion(self, trans_words, trans_opinion_words, from_to, opinion_from, opinion_to, has_neg):
        """Reverse the polarity of original opinion and return the new
        transformed opinion words.

        Args:
            trans_words: list
            trans_opinion_words: list
            from_to: list
            opinion_from: int
            opinion_to: int
            has_neg: bool
        Returns:
            trans_opinion_words: list
        """
        opinion_list = trans_words[opinion_from:opinion_to]
        opinion_tag = self.get_postag(trans_words, opinion_from, opinion_to)
        opinion_words = trans_words[opinion_from:opinion_to]
        opi = opinion_list[0]
        trans_opinion_word = None

        if has_neg and [opinion_from, opinion_to] not in from_to:
            trans_opinion_word = [opinion_from, opinion_to, self.untokenize(opinion_words)]
        elif [opinion_from, opinion_to] not in from_to:
            candidate = self.get_antonym_words(opi)
            refined_candidate = self.refine_candidate(trans_words, opinion_from, opinion_to, candidate)
            if len(refined_candidate) == 0:
                # negate the closest verb
                opi_tag2 = self.get_postag(trans_words, 0, -1)
                if opinion_tag[0][-1] in ['JJ', 'JJR', 'JJS'] or \
                        opinion_tag[0][-1] in ['NN', 'NNS', 'NNP', 'NNPS'] or \
                        opinion_tag[0][-1] == ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                    trans_opinion_word = [opinion_from, opinion_to, self.untokenize(['not', opi])]
                else:
                    dis = len(trans_words)
                    fidx = -1
                    for idx, (w, t) in enumerate(opi_tag2):
                        if abs(idx - opinion_from) < dis and w in \
                                ['is', 'was', 'are', 'were', 'am', 'being']:
                            dis = abs(idx - opinion_from)
                            fidx = idx
                    if fidx == -1:
                        trans_opinion_word = [opinion_from, opinion_to, self.untokenize(['not', opi])]
                    else:
                        trans_opinion_word = [fidx, fidx + 1, self.untokenize([opi_tag2[fidx][0], 'not'])]
            else:
                select = random.randint(0, len(refined_candidate) - 1)
                trans_opinion_word = [opinion_from, opinion_to, self.untokenize([refined_candidate[select]])]
        if trans_opinion_word is not None:
            trans_opinion_words.append(trans_opinion_word)
            from_to.append([opinion_from, opinion_to])

        return trans_opinion_words
