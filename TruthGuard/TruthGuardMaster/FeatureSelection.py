import DataPrep
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# Count Vectorizer
countV = CountVectorizer()
train_count = countV.fit_transform(DataPrep.train_news['Statement'].values)

print(countV)
print(train_count)

def get_count_vectorizer_stats():
    print("Vocab size:", train_count.shape)
    print("Vocabulary:", countV.vocabulary_)
    print("Feature names:", countV.get_feature_names()[:25])

# TF-IDF
tfidfV = TfidfTransformer()
train_tfidf = tfidfV.fit_transform(train_count)

def get_tfidf_stats():
    print("TF-IDF shape:", train_tfidf.shape)
    print("TF-IDF features:", train_tfidf.A[:10])

# TF-IDF with n-grams
tfidf_ngram = TfidfVectorizer(stop_words='english', ngram_range=(1, 4), use_idf=True, smooth_idf=True)

# POS Tagging
import nltk
nltk.download('treebank')
tagged_sentences = nltk.corpus.treebank.tagged_sents()
training_sentences = DataPrep.train_news['Statement']

# Feature Extraction for POS tagging
def features(sentence, index):
    word = sentence[index]
    return {
        'word': word,
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': word[0].upper() == word[0],
        'is_all_caps': word.upper() == word,
        'is_all_lower': word.lower() == word,
        'prefix-1': word[0],
        'prefix-2': word[:2],
        'prefix-3': word[:3],
        'suffix-1': word[-1],
        'suffix-2': word[-2:],
        'suffix-3': word[-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in word,
        'is_numeric': word.isdigit(),
        'capitals_inside': word[1:].lower() != word[1:]
    }

def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

# Using Word2Vec
import numpy as np
with open("glove.6B.50d.txt", "rb") as lines:
    w2v = {line.split()[0]: np.array(list(map(float, line.split()[1:]))) for line in lines}

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
