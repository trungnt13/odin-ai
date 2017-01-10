# ===========================================================================
# Popular encoding:
#  utf-8
#  ISO-8859-1
#  ascii
# ===========================================================================
from __future__ import print_function, division, absolute_import

import timeit
import string
from numbers import Number
from collections import OrderedDict, Iterator, Iterable, defaultdict
from abc import abstractmethod, ABCMeta
from six import add_metaclass

import numpy as np

from odin.utils import as_tuple, Progbar, pad_sequences, is_string
from odin.stats import freqcount
from multiprocessing import Pool, cpu_count

try:
    import spacy
except Exception as e:
    raise Exception('This module require "spaCy" for natural language processing.')

_nlp = {}
_stopword_list = []


def language(lang='en'):
    """Support language: 'en', 'de' """
    lang = lang.lower()
    if lang not in ('en', 'de'):
        raise ValueError('We only support languages: en-English, de-German.')
    if lang not in _nlp:
        _nlp[lang] = spacy.load(lang)
    return _nlp[lang]


def add_stopword(words):
    words = as_tuple(words, t=unicode)
    for w in words:
        _stopword_list.append(w)


def is_stopword(word, lang='en'):
    nlp = language(lang)
    # ====== check self-defined dictionary ====== #
    if word in _stopword_list:
        return True
    # ====== check in spacy dictionary ====== #
    if word not in nlp.vocab.strings:
        return False
    lexeme = nlp.vocab[nlp.vocab.strings[word]]
    return lexeme.is_stop


def is_oov(word, lang='en'):
    """ Check if a word is out of dictionary """
    nlp = language(lang)
    if word not in nlp.vocab.strings:
        return True
    return False


# ===========================================================================
# Text preprocessor
# ===========================================================================
@add_metaclass(ABCMeta)
class TextPreprocessor(object):
    """ A Preprocessor takes a string and return a preprocessed string
    a list of strings which represented a list of tokens.
    """

    @abstractmethod
    def preprocess(self, text):
        pass

    def __call__(self, text):
        if isinstance(text, (tuple, list)):
            return [self.preprocess(text) for t in text]
        else:
            return self.preprocess(text)


class CasePreprocessor(TextPreprocessor):

    def __init__(self, lower, keep_name=True, split=' '):
        super(CasePreprocessor, self).__init__()
        self.lower = lower
        self.split = split
        self.keep_name = keep_name

    def preprocess(self, text):
        if self.split is not None:
            text = text.split(' ')
            if self.lower:
                text = [t if self.keep_name and t.isupper()
                        else t.lower()
                        for t in text if len(t) > 0]
        elif self.lower:
            text = text.lower()
        return text


class TransPreprocessor(TextPreprocessor):
    """ Substitute a set of character to a new characters """

    def __init__(self, old='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 new=' '):
        super(TransPreprocessor, self).__init__()
        self.__str_trans = string.maketrans(old, new * len(old))
        new = None if len(new) == 0 else unicode(new)
        old = unicode(old)
        self.__uni_trans = dict((ord(char), new) for char in old)

    def preprocess(self, text):
        if isinstance(text, unicode):
            text = text.translate(self.__uni_trans)
        else:
            text = text.translate(self.__str_trans)
        return text.strip()


# ===========================================================================
# Simple Token filter
# ===========================================================================
@add_metaclass(ABCMeta)
class TokenFilter(object):
    """ A Filter return a "length > 0" token if the token is accepted
    and '' token otherwise.

    This allows not only token filtering but also token transforming.
    """

    @abstractmethod
    def filter(self, token, pos, is_stop, is_oov):
        pass

    def __call__(self, token, pos):
        return self.filter(token, pos)


class TYPEfilter(TokenFilter):
    """ Simplest form of filter
    is_alpha: alphabetic characters only (i.e. no space and digit)
    is_digit: consists of digits only
    is_ascii: all character must be ASCII
    is_title: the first letter if upper-case
    is_punct: punctuation

    * if any value True, the token with given type is accepted

    Paramters
    ---------
    keep_oov: bool
        if True keep all the "out-of-vocabulary" tokens, else ignore them

    """

    def __init__(self, is_alpha=False, is_digit=False, is_ascii=False,
                 is_title=False):
        super(TYPEfilter, self).__init__()
        self.is_alpha = is_alpha
        self.is_digit = is_digit
        self.is_ascii = is_ascii
        self.is_title = is_title

    def filter(self, token, pos):
        flags = [
            self.is_alpha and token.isalpha(),
            self.is_digit and token.isdigit(),
            self.is_title and token.istitle(),
            self.is_ascii and all(ord(c) < 128 for c in token)
        ]
        if any(flags):
            return token
        return ''


class POSfilter(TokenFilter):
    """ we use universal tag-of-speech for filtering token
    NOUN : noun
    PRON : pronoun (I, me, he, she, herself, you, it, that, etc.)
    PROPN: proper noun (name of individual person, place, ... spelled with an initial capital letter, e.g. Jane, London)
    ADJ  : adjective
    VERB : verb
    ADV  : adverb

    ADP  : adposition (prepositional, postpositional, and circumpositional phrases)
    AUX  : auxiliary
    DET  : determiner
    INTJ : interjection
    NUM  : numeral
    PART : particle
    PUNCT: punctuation
    SCONJ: subordinating conjunction
    SYM  : symbol
    X    : other

    """

    def __init__(self, NOUN=True, PRON=False, PROPN=True,
                 ADJ=True, VERB=False, ADV=True,
                 ADP=False, AUX=False, DET=False, INTJ=False,
                 NUM=False, PART=False, PUNCT=False,
                 SCONJ=False, SYM=False, X=False):
        super(POSfilter, self).__init__()
        pos = []
        if NOUN: pos.append('NOUN')
        if PRON: pos.append('PRON')
        if PROPN: pos.append('PROPN')
        if ADJ: pos.append('ADJ')
        if ADP: pos.append('ADP')
        if ADV: pos.append('ADV')
        if AUX: pos.append('AUX')
        if DET: pos.append('DET')
        if INTJ: pos.append('INTJ')
        if NUM: pos.append('NUM')
        if PART: pos.append('PART')
        if PUNCT: pos.append('PUNCT')
        if SCONJ: pos.append('SCONJ')
        if SYM: pos.append('SYM')
        if VERB: pos.append('VERB')
        if X: pos.append('X')
        self.pos = pos

    def filter(self, token, pos):
        """
        Paramters
        ---------
        tokens: list
            pass

        Return
        ------
        list of accepted tokens
        """
        if pos is None or pos in self.pos:
            return token
        return ''


# ===========================================================================
# Preprocessing data
# ===========================================================================
# static variables for multiprocessing
def _preprocess_func(doc):
    preprocessors = globals()['__preprocessors']
    filters = globals()['__filters']
    charlevel = globals()['__charlevel']
    lang = globals()['__lang']
    lemma = globals()['__lemma']
    stopwords = globals()['__stopwords']

    doc_tokens = []
    # preprocessing document
    for p in preprocessors:
        doc = p(doc)
    # auto split if the doc haven't been splitted
    if isinstance(doc, (str, unicode)):
        doc = doc.split(' ')
    # ====== start processing ====== #
    for token in doc:
        if len(token) > 0:
            # ignore stopwords if requred
            if not stopwords and is_stopword(token, lang):
                continue
            # check if token is accepted
            if filters is not None:
                for f in filters:
                    token = f(token, None)
            token = token.strip()
            # ignore token if it is removed
            if len(token) == 0: continue
            # normalize the token
            if lemma:
                pass
            # word-level dictionary
            if not charlevel:
                doc_tokens.append(token)
            # character-level dictionary
            else:
                for char in token:
                    doc_tokens.append(char)
    return doc_tokens


class Tokenizer(object):

    """
    Parameters
    ----------
    preprocessors:
        pass
    filters:
        pass
    nb_words: int
        pass
    char_level: bool
        pass
    stopwords: bool
        if True, all stopwords are accepted, otherwise, remove all stopwords
        during preprocessing in `fit` function.
    order: 'word' or 'doc'
        if 'word', order the dictionary by word frequency
        if 'doc', order the dictionary by docs frequency (i.e. the number
        of documents that the word appears in)
    """

    def __init__(self, nb_words=None,
                 char_level=False,
                 preprocessors=None,
                 filters=None,
                 stopwords=False,
                 lemmatization=True,
                 language='en',
                 batch_size=2048,
                 nb_threads=None,
                 order='word',
                 engine='odin'):
        # ====== internal states ====== #
        if engine not in ('spacy', 'odin'):
            raise ValueError('We only support 2 text processing engines: Spacy, or ODIN.')
        if order not in ('word', 'doc'):
            raise ValueError('The "order" argument must be "doc" or "word".')
        self.__engine = engine
        self.__order = order
        self.__longest_document = ['', 0]
        # ====== dictionary info ====== #
        self._nb_words = nb_words
        self.nb_docs = 0
        self.char_level = char_level
        self.language = language

        self._word_counts = defaultdict(int)
        # number of docs the word appeared
        self._word_docs = defaultdict(int)
        # actual dictionary used for embedding
        self._word_dictionary = OrderedDict()

        self.stopwords = stopwords
        self.lemmatization = lemmatization

        self.batch_size = batch_size
        self.nb_threads = (int(nb_threads) if nb_threads is not None
                           else cpu_count())
        # ====== filter and preprocessor ====== #
        self.filters = filters if filters is None else as_tuple(filters)
        if preprocessors is None:
            if engine == 'spacy':
                preprocessors = [TransPreprocessor()]
            elif engine == 'odin':
                preprocessors = [TransPreprocessor(), CasePreprocessor(lower=True)]
        elif not isinstance(preprocessors, (tuple, list)):
            preprocessors = [preprocessors]
        self.preprocessors = preprocessors

    def _refresh_dictionary(self):
        # sort the dictionary
        word_counts = self._word_counts.items() if self.__order == 'word' \
            else self._word_docs.items()
        word_counts.sort(key=lambda x: x[1], reverse=True)
        # create the ordered dictionary
        word_dictionary = OrderedDict()
        n = self.nb_words
        for i, (w, _) in enumerate(word_counts):
            if i > n:
                break
            word_dictionary[w] = i + 1
        word_dictionary[''] = 0
        word_dictionary[u''] = 0
        self._word_dictionary = word_dictionary
        return word_dictionary

    def _validate_texts(self, texts):
        if not isinstance(texts, Iterable) and \
        not isinstance(texts, Iterator) and \
        not is_string(texts):
            raise ValueError('texts must be an iterator, generator or a string.')
        if is_string(texts):
            texts = (texts,)
        return texts

    # ==================== properties ==================== #
    @property
    def nb_words(self):
        if self._nb_words is None or self._nb_words > len(self._word_counts):
            n = len(self._word_counts)
        else:
            n = int(self._nb_words)
        # always add the 0 words
        return n + 1

    @nb_words.setter
    def nb_words(self, value):
        if value != self._nb_words:
            self._nb_words = value
            self._refresh_dictionary()

    @property
    def longest_document(self):
        return self.__longest_document[0]

    @property
    def longest_document_length(self):
        return self.__longest_document[1]

    @property
    def summary(self):
        return {
            '#Words': len(self._word_counts),
            '#Docs': self.nb_docs,
            'DictSize': len(self.dictionary),
            'longest_docs': self.longest_document_length,
            'Top': [i[0] for i in self.top_k(n=12)]
        }

    @property
    def dictionary(self):
        return self._word_dictionary

    def __len__(self):
        return len(self._word_counts)

    def top_k(self, n=12):
        top = []
        count = self._word_counts if self.__order == 'word' else self._word_docs
        for i, w in enumerate(self._word_dictionary.iterkeys()):
            top.append((w, count[w]))
            if i >= n: break
        return top

    # ==================== methods ==================== #
    def _preprocess_docs_spacy(self, texts, vocabulary):
        def textit(texts):
            for t in texts:
                for p in self.preprocessors:
                    t = p(t)
                if isinstance(t, (tuple, list)):
                    t = ' '.join(t)
                yield t
        nlp = language(self.language)
        texts = textit(texts)
        for nb_docs, doc in enumerate(nlp.pipe(texts, n_threads=self.nb_threads,
                                               batch_size=self.batch_size)):
            doc_tokens = []
            for sent in doc.sents:
                for token in sent:
                    # check if token is accepted
                    token_POS = token.pos_
                    # ignore stopwords if requred
                    if not self.stopwords and is_stopword(token):
                        continue
                    # filtering
                    if self.filters is not None:
                        for f in self.filters:
                            token = f(token.orth_, token_POS)
                    # skip if token is removed by filter
                    if len(token) == 0: continue
                    # normalize the token
                    if self.lemmatization and token_POS != 'PROPN': # lemmalization
                        token = token.lemma_
                    else:
                        token = token.orth_
                    token = token.strip()
                    # word-level dictionary
                    if not self.char_level:
                        if vocabulary is None or token in vocabulary:
                            doc_tokens.append(token)
                    # character-level dictionary
                    else:
                        for char in token:
                            if vocabulary is None or token in vocabulary:
                                doc_tokens.append(char)
            yield nb_docs + 1, doc_tokens

    def _preprocess_docs_odin(self, texts, vocabulary):
        # ====== main processing ====== #
        def initializer(filters, preprocessors,
                        lang, lemma, charlevel, stopwords):
            globals()['__preprocessors'] = preprocessors
            globals()['__filters'] = filters
            globals()['__lang'] = lang
            globals()['__lemma'] = lemma
            globals()['__charlevel'] = charlevel
            globals()['__stopwords'] = stopwords

        nb_docs = 0
        pool = Pool(processes=self.nb_threads, initializer=initializer,
                    initargs=(self.filters, self.preprocessors, self.language,
                              self.lemmatization, self.char_level, self.stopwords))
        for doc in pool.imap_unordered(func=_preprocess_func, iterable=texts,
                                       chunksize=self.batch_size):
            nb_docs += 1
            if vocabulary is not None:
                doc = [token for token in doc if token in vocabulary]
            yield nb_docs, doc
        pool.close()
        pool.join()

    def fit(self, texts, vocabulary=None):
        """q
        Parameters
        ----------
        texts: iterator of unicode
            iterator, generator or list of unicode string.
        """
        texts = self._validate_texts(texts)
        word_counts = self._word_counts
        word_docs = self._word_docs
        # ====== pick engine ====== #
        if self.__engine == 'spacy':
            processor = self._preprocess_docs_spacy
        elif self.__engine == 'odin':
            processor = self._preprocess_docs_odin
        # ====== start processing ====== #
        prog = Progbar(target=1208)
        start_time = timeit.default_timer()
        for nb_docs, doc in processor(texts, vocabulary):
            total_docs_tokens = 0
            seen_words = {}
            # update words->count
            for token in doc:
                total_docs_tokens += 1
                word_counts[token] += 1
                # update words->doc
                if token not in seen_words:
                    seen_words[token] = 1
                    word_docs[token] += 1
            # save longest docs
            if total_docs_tokens > self.__longest_document[-1]:
                self.__longest_document = [doc, total_docs_tokens]
            # print progress
            prog.title = '[Training]#Doc:%d #Tok:%d' % (nb_docs, len(word_counts))
            prog.add(1)
            if prog.seen_so_far >= 0.8 * prog.target:
                prog.target = 1.2 * prog.target
        # ====== print summary of the process ====== #
        prog.target = nb_docs; prog.update(nb_docs)
        processing_time = timeit.default_timer() - start_time
        print('Processed %d-docs, %d-tokens in %f second.' %
            (nb_docs, len(word_counts), processing_time))
        self.nb_docs += nb_docs
        # ====== sorting ====== #
        self._refresh_dictionary()
        return self

    # ==================== transforming odin ==================== #
    def transform(self, texts, mode='seq', dtype='int32',
                  padding='pre', truncating='pre', value=0.,
                  end_document=None, maxlen=None,
                  token_not_found='ignore'):
        """
        Parameters
        ----------
        mode: 'binary', 'tfidf', 'count', 'freq', 'seq'
            'binary', abc
            'tfidf', abc
            'count', abc
            'freq', abc
            'seq', abc
        token_not_found: 'ignore', 'raise', a token string, an integer
            pass
        """
        # ====== check arguments ====== #
        texts = self._validate_texts(texts)
        # ====== check mode ====== #
        mode = str(mode)
        if mode not in ('seq', 'binary', 'count', 'freq', 'tfidf'):
            raise ValueError('The "mode" argument must be: "seq", "binary", '
                             '"count", "freq", or "tfidf".')
        # ====== check token_not_found ====== #
        if not isinstance(token_not_found, Number) and \
        not is_string(token_not_found) and \
        token_not_found not in ('ignore', 'raise'):
            raise ValueError('token_not_found can be: "ignore", "raise"'
                             ', an integer of token index, or a string '
                             'represented a token.')
        if token_not_found not in ('ignore', 'raise'):
            token_not_found = int(self.dictionary[token_not_found])
        elif isinstance(token_not_found, Number):
            token_not_found = int(token_not_found)
        # ====== pick engine ====== #
        if self.__engine == 'spacy':
            processor = self._preprocess_docs_spacy
        elif self.__engine == 'odin':
            processor = self._preprocess_docs_odin
        # ====== Initialize variables ====== #
        dictionary = self.dictionary
        results = []
        # ====== preprocess arguments ====== #
        if isinstance(end_document, str):
            end_document = dictionary.index(end_document)
        elif isinstance(end_document, Number):
            end_document = int(end_document)
        # ====== processing ====== #
        if hasattr(texts, '__len__'):
            target_len = len(texts)
            auto_adjust_len = False
        else:
            target_len = 1208
            auto_adjust_len = True
        prog = Progbar(target=target_len)
        for nb_docs, doc in processor(texts, vocabulary=None):
            # found the word in dictionary
            vec = []
            for x in doc:
                idx = dictionary.get(x, -1)
                if x >= 0: vec.append(idx)
                # not found the token in dictionary
                elif token_not_found == 'ignore': pass
                elif token_not_found == 'raise':
                    raise RuntimeError('Cannot find token: "%s" in dictionary' % x)
                elif isinstance(token_not_found, int):
                    vec.append(token_not_found)
            # append ending document token
            if end_document is not None:
                vec.append(end_document)
            # add the final results
            results.append(vec)
            # print progress
            prog.title = "[Transforming] %d docs" % nb_docs
            prog.add(1)
            if auto_adjust_len and prog.seen_so_far >= 0.8 * prog.target:
                prog.target = 1.2 * prog.target
        # end the process
        if auto_adjust_len:
            prog.target = nb_docs; prog.update(nb_docs)
        # ====== pad the sequence ====== #
        if mode == 'seq':
            maxlen = self.longest_document_length if maxlen is None \
                else int(maxlen)
            results = pad_sequences(results, maxlen=maxlen, dtype=dtype,
                                    padding=padding, truncating=truncating,
                                    value=value)
        else:
            X = np.zeros(shape=(len(results), self.nb_words))
            for i, j in enumerate(results):
                if mode == 'binary':
                    X[i, j] = 1
                elif mode == 'freq':
                    length = len(j)
                    count = freqcount(j)
                    for a, b in count.iteritems():
                        X[i, a] = b / float(length)
                elif mode == 'count':
                    count = freqcount(j)
                    for a, b in count.iteritems():
                        X[i, a] = b
                elif mode == 'tfidf':
                    count = freqcount(j)
                    for a, b in count.iteritems():
                        tf = 1 + np.log(b)
                        idf = np.log(1 + self.nb_docs /
                                     (1 + self._word_docs.get(j, 0)))
                        X[i][j] = tf * idf
            results = X
        return results

    def embed(self, vocabulary):
        """Any word not found in the vocabulary will be set to all-zeros"""
        ndim = len(vocabulary.itervalues().next())
        matrix = np.zeros(shape=(len(self.dictionary), ndim))
        for i, word in enumerate(self.dictionary):
            if word in vocabulary:
                matrix[i, :] = vocabulary[word]
        return matrix

    def __del__(self):
        pass
