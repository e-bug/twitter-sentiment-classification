import numpy as np
import re
import nltk
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import helpers
import hash_replacers

import pickle


# =========================================================================== #
# ============================ GLOABAL VARIABLES ============================ #
# =========================================================================== #

TEST_PATH = '../data/test_data.txt'

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

# LOAD PRE-TRAINED MODEL
print("Loading the pre-trained classifier...")
clf_logreg = pickle.load(open("models/clf_logreg.p", "rb"))


# LOAD TEST DATA
print("Loading the test data...")
ids, test = helpers.load_test_data(TEST_PATH)


# PRE-PROCESS DATA
print("Pre-processing the data...")

# Remove pound sign from hashtags
h_replacer = hash_replacers.RegexpReplacer()
for i,tweet in enumerate(test):
    test[i] = h_replacer.replace(tweet)

# Convert a collection of text documents to a matrix of token counts
vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    ngram_range = (1,3),
    max_df = 0.9261187281287935,
    min_df = 4
)
test_data_features = vectorizer.transform(test)

# Transform a count matrix to a normalized tf-idf representation
tfidf_transformer = TfidfTransformer()
test_data_features_tfidf = tfidf_transformer.transform(test_data_features)


# PREDICT LABELS
print("Predicting the labels...")
predicted_labels = clf_logreg.predict(test_data_features_tfidf)
predictions_list = [int(label) for label in predicted_labels]


# CREATE SUBMISSION FILE
print("Creating submission file in results/ folder")
helpers.create_csv_submission(ids, predictions_list, 'results/submission.csv')
