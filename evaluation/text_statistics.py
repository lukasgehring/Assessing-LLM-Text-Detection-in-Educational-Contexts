import nltk
import textstat

from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def statistical_analysis(corpus: DataFrame, column:str="text") -> DataFrame:
    """
    Compute statistical analysis metrics for a text corpus. The metrics include: 'word_count', 'sentence_count', 'avg_sentence_length', 'unique_word_count', 'lexical_diversity' and 'flesch_reading_ease'.

    :param corpus: Text corpus.
    :param column: Column name of the text corpus.

    :return: DataFrame with computed statistical analysis metrics.
    """
    corpus = apply_avg_sentence_length(corpus, column=column, drop_intermediate=False)
    corpus = apply_lexical_diversity(corpus, column=column, drop_intermediate=False)
    corpus = apply_flesch_reading_ease(corpus, column=column)

    return corpus

def apply_word_count(corpus: DataFrame, column:str="text") -> DataFrame:
    """
    Compute word count metric for a text corpus.

    :param corpus: Text corpus.
    :param column: Column name of the text corpus.
    :return: DataFrame with computed word count metric 'word_count'.
    """
    corpus['word_count'] = corpus[column].apply(lambda x: len(nltk.word_tokenize(x)))
    return corpus

def apply_sentence_count(corpus: DataFrame, column:str="text") -> DataFrame:
    """
    Compute sentence count metric for a text corpus.

    :param corpus: Text corpus.
    :param column: Column name of the text corpus.
    :return: DataFrame with computed sentence count metric 'sentence_count'.
    """
    corpus['sentence_count'] = corpus[column].apply(lambda x: len(nltk.sent_tokenize(x)))
    return corpus

def apply_avg_sentence_length(corpus: DataFrame, column:str="text", drop_intermediate:bool=True) -> DataFrame:
    """
    Compute average sentence length metric for a text corpus.

    :param corpus: Text corpus.
    :param column: Column name of the text corpus.
    :param drop_intermediate: If `True`, intermediate metrics added for computation are dropped.
    :return: DataFrame with computed average sentence length metric 'avg_sentence_length'.
    """
    drop_columns = []

    if 'word_count' not in corpus.columns:
        drop_columns.append('word_count')
        corpus = apply_word_count(corpus, column=column)

    if 'sentence_count' not in corpus.columns:
        drop_columns.append('sentence_count')
        corpus = apply_sentence_count(corpus,column=column)

    corpus['avg_sentence_length'] = corpus['word_count'] / corpus['sentence_count']

    if drop_intermediate and drop_columns:
        corpus = corpus.drop(columns=drop_columns)

    return corpus

def apply_unique_word_count(corpus: DataFrame, column:str="text") -> DataFrame:
    """
    Compute unique word count metric for a text corpus.

    :param corpus: Text corpus.
    :param column: Column name of the text corpus.
    :return: DataFrame with computed unique word count metric 'unique_word_count'.
    """
    corpus['unique_word_count'] = corpus[column].apply(lambda x: len(set(nltk.word_tokenize(x))))
    return corpus

def apply_lexical_diversity(corpus: DataFrame, column:str="text", drop_intermediate:bool=True) -> DataFrame:
    """
    Compute lexical diversity metric for a text corpus.

    :param corpus: Text corpus.
    :param column: Column name of the text corpus.
    :param drop_intermediate: If `True`, intermediate metrics added for computation are dropped.
    :return: DataFrame with computed lexical diversity metric 'lexical_diversity'.
    """
    drop_columns = []

    if 'word_count' not in corpus.columns:
        drop_columns.append('word_count')
        corpus = apply_word_count(corpus, column=column)

    if 'unique_word_count' not in corpus.columns:
        drop_columns.append('unique_word_count')
        corpus = apply_unique_word_count(corpus, column=column)

    corpus['lexical_diversity'] = corpus['unique_word_count'] / corpus['word_count']

    if drop_intermediate and drop_columns:
        corpus = corpus.drop(columns=drop_columns)

    return corpus

def apply_flesch_reading_ease(corpus: DataFrame, column:str="text") -> DataFrame:
    """
    Compute flesch-reading ease metric for a text corpus.

    :param corpus: Text corpus.
    :param column: Column name of the text corpus.
    :return: DataFrame with computed flesch-reading ease metric 'flesch_reading_ease'.
    """
    corpus['flesch_reading_ease'] = corpus[column].apply(textstat.textstat.flesch_reading_ease)
    return corpus

def apply_cosine_similarity(corpus: DataFrame, x_column:str, y_column:str, vectorizer = TfidfVectorizer()) -> DataFrame:
    """
    Compute cosine similarity metric between two text corpora.

    :param corpus: DataFrame with text corpora.
    :param x_column: Column name of the first text corpus.
    :param y_column: Column name of the second text corpus.
    :param vectorizer: Vectorizer.

    :return: DataFrame with computed cosine similarity metric 'cosine_similarity'.
    """

    def cosine_similarity_text(row):
        vectorized = vectorizer.fit_transform([row[x_column], row[y_column]])
        similarity = cosine_similarity(vectorized[0:1], vectorized[1:2])[0][0]
        return similarity

    corpus["cosine_similarity"] = corpus.apply(cosine_similarity_text, axis=1, vectorizer=vectorizer)
    return corpus

