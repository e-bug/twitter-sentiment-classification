import numpy as np
import re
import nltk
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

import helpers
import hash_replacers

import pickle


# =========================================================================== #
# ============================ GLOABAL VARIABLES ============================ #
# =========================================================================== #

STEMMER = EnglishStemmer(ignore_stopwords=True)


# =========================================================================== #
# ================================ FUNCTIONS ================================ #
# =========================================================================== #

def tokenize(text):
    """
    Tokenize passed text by replacing repeating letters and stemming.
    :param text: string to be tokenized
    :return: tokenized and pre-processed text
    """
    # Tokenize
    tokens = nltk.word_tokenize(text)    
    
    # Replace two or more consecutive equal characters with two occurrences
    replaced_tokens = [helpers.replace_two_or_more(w) for w in tokens]
    
    # Stem
    stemmed_tokens = helpers.stem_tokens(replaced_tokens, STEMMER)
    
    return stemmed_tokens


# =========================================================================== #
# =================================== MAIN ================================== #
# =========================================================================== #

# LOAD DATA	
print("Loading the training data...")
pos_tr_full = helpers.load_data('../data/train_pos_full.txt')
neg_tr_full = helpers.load_data('../data/train_neg_full.txt')
X_train_full = np.append(pos_tr_full, neg_tr_full)


# LABEL DATA
pos_labels_full = np.ones(len(pos_tr_full))
neg_labels_full = np.zeros(len(neg_tr_full)) - 1
y_train_full = np.append(pos_labels_full, neg_labels_full)


# PRE-PROCESS DATA
print("Pre-processing the data...")

# Remove pound sign from hashtags
h_replacer = hash_replacers.RegexpReplacer()
for i,tweet in enumerate(X_train_full):
    X_train_full[i] = h_replacer.replace(tweet)

# Convert a collection of text documents to a matrix of token counts
vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    ngram_range = (1,3),
    max_df = 0.9261187281287935,
    min_df = 4
)
corpus_data_features = vectorizer.fit_transform(X_train_full)

# Transform a count matrix to a normalized tf-idf representation
tfidf_transformer = TfidfTransformer()
corpus_data_features_tfidf = tfidf_transformer\
                              .fit_transform(corpus_data_features)


# TRAIN CLASSIFIER
print("Training Logistic Regression classifier...")
clf_logreg = LogisticRegression(max_iter=100, n_jobs=-1, C=3.41)\
              .fit(corpus_data_features_tfidf, y_train_full)


# SAVE MODEL
print("Saving the trained classifier in models/ folder...")
pickle.dump(clf_logreg, open('models/clf_logreg.p', 'wb'))
