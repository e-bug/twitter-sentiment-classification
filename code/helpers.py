# -*- coding: utf-8 -*-
"""some functions for help."""

import numpy as np
import csv

import random
import re


def read_txt(path):
    """
    Read text file from path.
    :param path: path of the file to read
    :return: file splitted by lines
    """
    with open(path, "r", encoding="utf8") as f:
        return f.read().splitlines()


def load_data(path_dataset, subsample_size=0):
    """
    Load data in text format, one rating per line.
    :param path_dataset: path of the file to read
    :param subsample_size: number of lines to read (Default: 0, all lines)
    :return: file splitted by lines
    """
    data = read_txt(path_dataset)
    
    if subsample_size:
        random_indices = random.sample(range(len(data)), 
                                       int(round(subsample_size)))
        rand_smpl = [data[i] for i in random_indices]
        data = rand_smpl

    return data


def stem_tokens(tokens, stemmer):
    """
    Stem the passed tokens using the specified stemmer.
    :param tokens: tokens to be stemmed
    :param stemmer: instance of a stemmer from NLTK
    :return: stemmed tokens
    """
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))

    return stemmed


def replace_two_or_more(s):
    """
    Look for 2 or more repetitions of character and replace with the character itself.
    :param s: a string specifying a word
    :return: word with replaced repetitions
    """
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    
    return pattern.sub(r"\1\1", s)


def load_test_data(path_dataset):
    """
    Load data in text format, one (id,tweet) per line.
    :param path_dataset: path of the file to read
    :return ids, tweets: lists of ids and associated tweets
    """
    id_tweet_list = load_data(path_dataset)
    
    ids = []
    tweets = []
    for id_tweet in id_tweet_list:
        id_, tweet = id_tweet.split(sep=',', maxsplit=1)
        ids.append(id_)
        tweets.append(tweet)
    
    return ids, tweets


def create_csv_submission(ids, y_pred, name):
    """
    Create an output file in csv format for submission to kaggle.
    :param ids: event ids associated with each prediction
    :param y_pred: predicted class labels
    :param name: string name of .csv output file to be created
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})



