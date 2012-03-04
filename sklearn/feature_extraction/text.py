# -*- coding: utf-8 -*-
# Authors: Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
#          Robert Layton <robertlayton@gmail.com>
#
# License: BSD Style.
"""
The :mod:`sklearn.feature_extraction.text` submodule gathers utilities to
build feature vectors from text documents.
"""

import re
import unicodedata
from operator import itemgetter

import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator, TransformerMixin
from ..preprocessing import normalize
from ..utils.fixes import Counter
from .stop_words import ENGLISH_STOP_WORDS


def strip_accents_unicode(s):
    """Transform accentuated unicode symbols into their simple counterpart

    Warning: the python-level loop and join operations make this
    implementation 20 times slower than the strip_accents_ascii basic
    normalization.

    See also
    --------
    strip_accents_ascii
        Remove accentuated char for any unicode symbol that has a direct
        ASCII equivalent.
    """
    return u''.join([c for c in unicodedata.normalize('NFKD', s)
                     if not unicodedata.combining(c)])


def strip_accents_ascii(s):
    """Transform accentuated unicode symbols into ascii or nothing

    Warning: this solution is only suited for languages that have a direct
    transliteration to ASCII symbols.

    See also
    --------
    strip_accents_unicode
        Remove accentuated char for any unicode symbol.
    """
    nkfd_form = unicodedata.normalize('NFKD', s)
    only_ascii = nkfd_form.encode('ASCII', 'ignore')
    return only_ascii


def strip_tags(s):
    """Basic regexp based HTML / XML tag stripper function

    For serious HTML/XML preprocessing you should rather use an external
    library such as lxml or BeautifulSoup.
    """
    return re.compile(ur"<([^>]+)>", flags=re.UNICODE).sub(u" ", s)


def _check_stop_list(stop):
    if stop == "english":
        return ENGLISH_STOP_WORDS
    elif isinstance(stop, str) or isinstance(stop, unicode):
        raise ValueError("not a built-in stop list: %s" % stop)
    else:               # assume it's a collection
        return stop


class CountVectorizer(BaseEstimator):
    """Convert a collection of raw documents to a matrix of token counts

    This implementation produces a sparse representation of the counts using
    scipy.sparse.coo_matrix.

    If you do not provide an a-priori dictionary and you do not use an analyzer
    that does some kind of feature selection then the number of features will
    be equal to the vocabulary size found by analysing the data. The default
    analyzer does simple stop word filtering for English.

    Parameters
    ----------
    input: string {'filename', 'file', 'content'}
        If filename, the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have 'read' method (file-like
        object) it is called to fetch the bytes in memory.

        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    charset: string
        If bytes or files are given to analyze, this charset is used to
        decode.

    charset_error: {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `charset`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    tokenize: string, {'word', 'char'}
        Whether the feature should be made of word or character n-grams.

    min_n: integer
        The lower boundary of the range of n-values for different n-grams to be
        extracted.

    max_n: integer
        The upper boundary of the range of n-values for different n-grams to be
        extracted. All values of n such that min_n <= n <= max_n will be used.

    strip_accents: string {'ascii', 'unicode'} or False
        If False, accentuated chars are kept as this.

        If 'ascii', accentuated chars are converted to there ascii non
        accentuated equivalent: fast processing but only suitable for roman
        languages.

        If 'unicode', accentuated chars are converted to there non accentuated
        equivalent: slower that 'ascii' but works for any language.

    stop_words: string {'english'}, list, or None (default)
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned is currently the only
        supported string value.

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    token_pattern: string
        Regular expression denoting what constitutes a "token", only used
        if `tokenize == 'word'`. The default regexp select tokens of 2
        or more letters characters (punctuation is completely ignored
        and always treated as a token separator).

    max_df : float in range [0.0, 1.0], optional, 1.0 by default
        When building the vocabulary ignore terms that have a term frequency
        strictly higher than the given threshold (corpus specific stop words).

        This parameter is ignored if vocabulary is not None.

    max_features : optional, None by default
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.

    binary: boolean, False by default.
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    dtype: type, optional
        Type of the matrix returned by fit_transform() or transform().
    """

    _white_spaces = re.compile(ur"\s\s+")

    def __init__(self, input='content', charset='utf-8',
                 charset_error='strict', strip_accents='ascii',
                 strip_tags=False, lowercase=True, tokenize='word',
                 stop_words=None, token_pattern=ur"\b\w\w+\b",
                 min_n=1, max_n=1, max_df=1.0, max_features=None,
                 fixed_vocabulary=None, binary=False, dtype=long):
        self.input = input
        self.charset = charset
        self.charset_error = charset_error
        self.strip_accents = strip_accents
        self.strip_tags = strip_tags
        self.lowercase = lowercase
        self.min_n = min_n
        self.max_n = max_n
        self.tokenize = tokenize
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_df = max_df
        self.max_features = max_features
        if (fixed_vocabulary is not None
            and not hasattr(fixed_vocabulary, 'get')):
            fixed_vocabulary = dict(
                (t, i) for i, t in enumerate(fixed_vocabulary))
        self.fixed_vocabulary = fixed_vocabulary
        self.binary = binary
        self.dtype = dtype

    def _decode(self, doc):
        if self.input == 'filename':
            doc = open(doc, 'rb').read()

        elif self.input == 'file':
            doc = doc.read()

        if isinstance(doc, bytes):
            doc = doc.decode(self.charset, self.charset_error)
        return doc

    def _word_tokenize(self, text_document, token_pattern, stop_words=None):
        """Tokenize text_document into a sequence of word n-grams"""
        tokens = token_pattern.findall(text_document)

        # handle token n-grams
        if self.min_n != 1 or self.max_n != 1:
            original_tokens = tokens
            tokens = []
            n_original_tokens = len(original_tokens)
            for n in xrange(self.min_n,
                            min(self.max_n + 1, n_original_tokens + 1)):
                for i in xrange(n_original_tokens - n + 1):
                    tokens.append(u" ".join(original_tokens[i: i + n]))

        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        return tokens

    def _char_tokenize(self, text_document):
        """Tokenize text_document into a sequence of character n-grams"""
        # normalize white spaces
        text_document = self._white_spaces.sub(u" ", text_document)

        text_len = len(text_document)
        ngrams = []
        for n in xrange(self.min_n, min(self.max_n + 1, text_len + 1)):
            for i in xrange(text_len - n + 1):
                ngrams.append(text_document[i: i + n])
        return ngrams

    def build_preprocessor(self):
        """Return a function to preprocess the text before tokenization"""
        noop = lambda x: x

        # accent stripping
        if not self.strip_accents:
            strip_accents = noop
        elif hasattr(self.strip_accents, '__call__'):
            strip_accents = self.strip_accents
        elif self.strip_accents == 'ascii':
            strip_accents = strip_accents_ascii
        elif self.strip_accents == 'unicode':
            strip_accents = strip_accents_unicode
        else:
            raise ValueError('Invalid value for "strip_accents": %s' %
                             self.strip_accents)

        # tags removal
        if hasattr(self.strip_tags, '__call__'):
            tags = self.strip_tags
        elif self.strip_tags:
            tags = strip_tags
        else:
            tags = noop

        if self.lowercase:
            return lambda x: strip_accents(tags(x.lower()))
        else:
            return lambda x: strip_accents(tags(x))

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        preprocess = self.build_preprocessor()

        if self.tokenize == 'char':
            return lambda doc: self._char_tokenize(
                preprocess(self._decode(doc)))

        elif self.tokenize == 'word':
            token_pattern = re.compile(self.token_pattern)
            stop_words = _check_stop_list(self.stop_words)

            return lambda doc: self._word_tokenize(
                preprocess(self._decode(doc)), token_pattern, stop_words)

        else:
            raise ValueError('%s is not a valid tokenization scheme' %
                             self.tokenize)

    def _term_count_dicts_to_matrix(self, term_count_dicts):
        i_indices = []
        j_indices = []
        values = []
        if self.fixed_vocabulary is not None:
            vocabulary = self.fixed_vocabulary
        else:
            vocabulary = self.vocabulary_

        for i, term_count_dict in enumerate(term_count_dicts):
            for term, count in term_count_dict.iteritems():
                j = vocabulary.get(term)
                if j is not None:
                    i_indices.append(i)
                    j_indices.append(j)
                    values.append(count)
            # free memory as we go
            term_count_dict.clear()

        shape = (len(term_count_dicts), max(vocabulary.itervalues()) + 1)
        spmatrix = sp.coo_matrix((values, (i_indices, j_indices)),
                                 shape=shape, dtype=self.dtype)
        if self.binary:
            spmatrix.data[:] = 1
        return spmatrix

    def fit(self, raw_documents, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents

        Parameters
        ----------
        raw_documents: iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return the count vectors

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        raw_documents: iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        vectors: array, [n_samples, n_features]
        """
        if self.fixed_vocabulary is not None:
            # not need to fit anything, directly perform the transformation
            return self.transform(raw_documents)

        self.vocabulary_ = {}
        # result of document conversion to term count dicts
        term_counts_per_doc = []
        term_counts = Counter()

        # term counts across entire corpus (count each term maximum once per
        # document)
        document_counts = Counter()

        max_df = self.max_df
        max_features = self.max_features

        analyze = self.build_analyzer()

        # TODO: parallelize the following loop with joblib?
        # (see XXX up ahead)
        for doc in raw_documents:
            term_count_current = Counter(analyze(doc))
            term_counts.update(term_count_current)

            if max_df < 1.0:
                document_counts.update(term_count_current.iterkeys())

            term_counts_per_doc.append(term_count_current)

        n_doc = len(term_counts_per_doc)

        # filter out stop words: terms that occur in almost all documents
        if max_df < 1.0:
            max_document_count = max_df * n_doc
            stop_words = set(t for t, dc in document_counts.iteritems()
                               if dc > max_document_count)
        else:
            stop_words = set()

        # list the terms that should be part of the vocabulary
        if max_features is None:
            terms = set(term_counts) - stop_words
        else:
            # extract the most frequent terms for the vocabulary
            terms = set()
            for t, tc in term_counts.most_common():
                if t not in stop_words:
                    terms.add(t)
                if len(terms) >= max_features:
                    break

        # store the learned stop words to make it easier to debug the value of
        # max_df
        self.max_df_stop_words_ = stop_words

        # store map from term name to feature integer index: we sort the term
        # to have reproducible outcome for the vocabulary structure: otherwise
        # the mapping from feature name to indices might depend on the memory
        # layout of the machine. Furthermore sorted terms might make it
        # possible to perform binary search in the feature names array.
        self.vocabulary_ = dict(((t, i) for i, t in enumerate(sorted(terms))))

        # the term_counts and document_counts might be useful statistics, are
        # we really sure want we want to drop them? They take some memory but
        # can be useful for corpus introspection
        return self._term_count_dicts_to_matrix(term_counts_per_doc)

    def transform(self, raw_documents):
        """Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided in the constructor.

        Parameters
        ----------
        raw_documents: iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        vectors: sparse matrix, [n_samples, n_features]
        """
        if self.fixed_vocabulary is None and not hasattr(self, 'vocabulary_'):
            raise ValueError("Vocabulary wasn't fitted or is empty!")

        # raw_documents can be an iterable so we don't know its size in
        # advance

        # XXX @larsmans tried to parallelize the following loop with joblib.
        # The result was some 20% slower than the serial version.
        analyze = self.build_analyzer()
        term_counts_per_doc = [Counter(analyze(doc)) for doc in raw_documents]
        return self._term_count_dicts_to_matrix(term_counts_per_doc)

    def inverse_transform(self, X):
        """Return terms per document with nonzero entries in X.

        Parameters
        ----------
        X : {array, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        X_inv : list of arrays, len = n_samples
            List of arrays of terms.
        """
        if sp.isspmatrix_coo(X):  # COO matrix is not indexable
            X = X.tocsr()
        elif not sp.issparse(X):
            # We need to convert X to a matrix, so that the indexing
            # returns 2D objects
            X = np.asmatrix(X)
        n_samples = X.shape[0]

        terms = np.array(self.vocabulary_.keys())
        indices = np.array(self.vocabulary_.values())
        inverse_vocabulary = terms[np.argsort(indices)]

        return [inverse_vocabulary[X[i, :].nonzero()[1]].ravel()
                for i in xrange(n_samples)]

    def get_vocabulary(self):
        """Dict mapping from string feature name to feature integer index

        If fixed_vocabulary was passed to the constructor, it is returned,
        otherwise, the `vocabulary_` attribute built during fit is returned
        instead.
        """
        if self.fixed_vocabulary is not None:
            return self.fixed_vocabulary
        else:
            return getattr(self, 'vocabulary_', {})

    def get_feature_names(self):
        """Array mapping from feature integer indicex to feature name"""
        vocabulary = self.get_vocabulary()
        return np.array([t for t, i in sorted(vocabulary.iteritems(),
                                              key=itemgetter(1))])


class TfidfTransformer(BaseEstimator, TransformerMixin):
    """Transform a count matrix to a normalized tf or tf–idf representation

    Tf means term-frequency while tf–idf means term-frequency times inverse
    document-frequency. This is a common term weighting scheme in information
    retrieval, that has also found good use in document classification.

    The goal of using tf–idf instead of the raw frequencies of occurrence of a
    token in a given document is to scale down the impact of tokens that occur
    very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus.

    In the SMART notation used in IR, this class implements several tf–idf
    variants. Tf is always "n" (natural), idf is "t" iff use_idf is given,
    "n" otherwise, and normalization is "c" iff norm='l2', "n" iff norm=None.

    Parameters
    ----------
    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.

    use_idf : boolean, optional
        Enable inverse-document-frequency reweighting.

    smooth_idf : boolean, optional
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : boolean, optional
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Notes
    -----
    **References**:

    .. [Yates2011] `R. Baeza-Yates and B. Ribeiro-Neto (2011). Modern
                   Information Retrieval. Addison Wesley, pp. 68–74.`

    .. [MSR2008] `C.D. Manning, H. Schütze and P. Raghavan (2008). Introduction
                 to Information Retrieval. Cambridge University Press,
                 pp. 121–125.`
    """

    def __init__(self, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.idf_ = None

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights)

        Parameters
        ----------
        X: sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        if self.use_idf:
            n_samples, n_features = X.shape
            df = np.bincount(X.nonzero()[1])
            if df.shape[0] < n_features:
                # bincount might return fewer bins than there are features
                df = np.concatenate([df, np.zeros(n_features - df.shape[0])])
            df += int(self.smooth_idf)
            self.idf_ = np.log(float(n_samples) / df)

        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf–idf representation

        Parameters
        ----------
        X: sparse matrix, [n_samples, n_features]
            a matrix of term/token counts

        Returns
        -------
        vectors: sparse matrix, [n_samples, n_features]
        """
        X = sp.csr_matrix(X, dtype=np.float64, copy=copy)
        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            expected_n_features = self.idf_.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            d = sp.lil_matrix((n_features, n_features))
            d.setdiag(self.idf_)
            # *= doesn't work
            X = X * d

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X


class Vectorizer(BaseEstimator):
    """Convert a collection of raw documents to a matrix of TF-IDF features.

    Equivalent to CountVectorizer followed by TfidfTransformer.

    See also
    --------
    CountVectorizer
        Tokenize the documents and count the occurrences of token and return
        them as a sparse matrix

    TfidfTransformer
        Apply Term Frequency Inverse Document Frequency normalization to a
        sparse matrix of occurrence counts.

    """

    def __init__(self, max_df=1.0,
                 max_features=None, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.tc = CountVectorizer(max_df=max_df,
                                  max_features=max_features,
                                  dtype=np.float64)
        self.tfidf = TfidfTransformer(norm=norm, use_idf=use_idf,
                                      smooth_idf=smooth_idf,
                                      sublinear_tf=sublinear_tf)

    def fit(self, raw_documents):
        """Learn a conversion law from documents to array data"""
        X = self.tc.fit_transform(raw_documents)
        self.tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn the representation and return the vectors.

        Parameters
        ----------
        raw_documents: iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        vectors: array, [n_samples, n_features]
        """
        X = self.tc.fit_transform(raw_documents)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self.tfidf.fit(X).transform(X, copy=False)

    def transform(self, raw_documents, copy=True):
        """Transform raw text documents to tf–idf vectors

        Parameters
        ----------
        raw_documents: iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        vectors: sparse matrix, [n_samples, n_features]
        """
        X = self.tc.transform(raw_documents)
        return self.tfidf.transform(X, copy)

    def inverse_transform(self, X):
        """Return terms per document with nonzero entries in X.

        Parameters
        ----------
        X : {array, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        X_inv : list of arrays, len = n_samples
            List of arrays of terms.
        """
        return self.tc.inverse_transform(X)

    def build_analyzer(self):
        return self.tc.build_analyzer()

    def get_vocabulary(self):
        return self.tc.get_vocabulary()

    def get_feature_names(self):
        return self.tc.get_feature_names()

    vocabulary_ = property(lambda self: self.tc.vocabulary_)
