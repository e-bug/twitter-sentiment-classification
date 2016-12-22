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
print("Pre-processing and feature representation... " 
      "This takes around 40 minutes!")

# Remove pound sign from hashtags
h_replacer = hash_replacers.RegexpReplacer()
for i,tweet in enumerate(X_train_full):
    X_train_full[i] = h_replacer.replace(tweet)

# Convert collection of text documents to a matrix of token counts
vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    ngram_range = (1,3),
    max_df = 0.9261187281287935,
    min_df = 4
)
corpus_data_fitted = vectorizer.fit(X_train_full)
pickle.dump(corpus_data_fitted.vocabulary_, 	# save vocabulary for future
            open('models/vocabulary.p', 'wb'))	# predictions
corpus_data_features = corpus_data_fitted.transform(X_train_full)


# Transform count matrix to a normalized tf-idf representation
tfidf_transformer = TfidfTransformer()
corpus_data_tfidf_fitted = tfidf_transformer.fit(corpus_data_features)
pickle.dump(corpus_data_tfidf_fitted, 		# save fitted Tfidf for future
            open('models/corpus_data_tfidf_fitted.p', 'wb'))	# predictions
corpus_data_features_tfidf = corpus_data_tfidf_fitted\
                              .transform(corpus_data_features)


# TRAIN CLASSIFIER
print("Training Logistic Regression classifier... "
      "This takes around 10 minutes!")
clf_logreg = LogisticRegression(max_iter=100, n_jobs=-1, C=3.41)\
              .fit(corpus_data_features_tfidf, y_train_full)


# SAVE MODEL
print("Saving the trained classifier in models/ folder...")
pickle.dump(clf_logreg, open('models/clf_logreg.p', 'wb'))

print("Model successfully trained!")
