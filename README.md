# ML Project 2: Twitter Sentiment Classification
Repository for the second project in Machine Learning, EPFL 2016/2017. <br><br>
Project: Text classification<br>
Authors: Emanuele Bugliarello, Manik Garg, Zander Harteveld<br>
Team: Memory Error <br>
<hr>
_For an easier examination by the EPFL TAs, this content is also available on [GitHub](http://github.com/e-bug/pcml-project2 "GitHub repository for Memory Error's code") (publicly after the deadline)._<br>
<hr>
## Overview
This repository contains the material shipped on December 22 and consists of the following folders:
- `data`: contains the [Twitter data files from Kaggle](https://inclass.kaggle.com/c/epfml-text/data)
- `code`: contains the Python files used to train the model and generate new predictions. Details about the files are available in the [README](code/README.md) inside the `code` folder.
- `report`: contains the submitted report and the files used to generate it.

## Dependencies
The code is written in `Python3`, that you can download from [here](https://www.python.org/downloads/release/python-352/) (we recommend installing a virtual environment such as [Anaconda](https://www.continuum.io/downloads) that already comes with many libraries).<br>
The libraries required are:
- NumPy (>= 1.6.1): you can install it by typying `pip install -U numpy` on the terminal (it is included with Anaconda).
- NLTK (3.0): you can install it by typying `pip install -U nltk` on the terminal.
- NLTK packages: you can download alll the packages of NLTK by typying `python` on the terminal. Then:
  ```python
  import nltk
  nltk.download('all')
  ```
  It will automatically install all the packages of NLTK. Note that it takes a lot of time to download the `panlex_lite` package but you can stop the execution because the packages needed by our scripts will have been already installed.

- SciPy (>=0.9): you can install it by typying `pip install -U scipy` on the terminal (it is in included with Anaconda).
- scikit-learn (0.18.1): you can install it by typying `pip install -U scikit-learn`, or `conda install scikit-learn` if you use Anaconda, on the terminal.

## Methodology
The final model consists of a Logistic Regression classifier. <br>
We apply the following pre-processing steps before feeding the data into the classifier:
  1. Remove the pound sign (#) in fron of words
  2. Stem words (by using `EnglishStemmer` from `nltk.stem.snowball`)
  3. Replace two or more consecutive repetitions of a letter with two of the same
  
We then convert the collection of text documents to a matrix of token counts. We do this with `CountVectorizer` from `sklearn.feature_extraction.text`, with the following hyperparameters:
 - analyzer = 'word'
 - tokenizer = tokenize (function that tokenizes the text by applying the pre-processing steps described above)
 - lowercase = True
 - ngram_range = (1,3)
 - max_df = 0.9261187281287935 
 - min_df = 4 
 
After that, we transform the count matrix to a normalized tf-idf representation with `TfidfTransformer`from `sklearn.feature_extraction.text`.
 
Finally, we feed this representation into the Logistic Regression classifier from `sklearn.linear_model`, parameterized with the following value of the inverse of regularization strength:
- C = 3.41

## Kaggle result reproduction
In order to generate the top Kaggle submission, please ensure all Python [requirements](#dependencies) are installed and then run:
```sh
cd code
python run.py
```
This makes use of the pre-trained classifier available in the `code/models` folder to predict labels for new tweets and store them in a `.csv` file in the `code/results` folder. The default test data file is `data/test_data.txt` (the one provided for the Kaggle competition) but it can be easily changed in `code/run.py`.

## Training from scratch
You can train the classifier that we use for the top Kaggle submission. To do:
  1. Ensure all Python [requirements](#dependencies) are installed
  2. Ensure the [Twitter data files from Kaggle](https://inclass.kaggle.com/c/epfml-text/data) are in the `data/` folder.
  3. Run:
  ```sh
  cd code
  python train.py
  ```
  
  This file makes use of `data/train_pos_full.txt` and `data/train_neg_full.txt` (data files from the Kaggle competition) as the training sets and creates a model in the `code/models` folder.
  The time needed to run it is between 30 and 45 minutes: 20-30 minutes for pre-processing and around 10 minutes for fitting the classifier (depending on your machine).
 
You can then predict labels for new data as described in the [previous section](#kaggle-result-reproduction).
