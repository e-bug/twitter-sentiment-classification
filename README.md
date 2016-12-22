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
- [`data`](data/): Contains the [Twitter data files from Kaggle](https://inclass.kaggle.com/c/epfml-text/data)
- [`code`](code/): Contains the Python files used to train the model and generate new predictions. Details about the files are available in the [README](code/README.md) inside the `code` folder.
- [`report`](report/): Contains the [submitted report](report/memory_error_text_classification.pdf) and the files used to generate it.

### Dependencies

### Data preparation

### Methodology

### Model selection

## Kaggle result reproduction
In order to generate the top Kaggle submission, please ensure all Python [requirements](#dependencies) are installed and then run:
```sh
cd code
python run.py
```
This makes use of the pre-trained classifier available in the [`models`](code/models/) folder to predict labels for new tweets and store them in a `.csv` file in the [`results`](code/results/) folder. The default `test data` is the one provided for the Kaggle competition, but can be easily changed in [`run.py`](code/run.py).
## Training from scratch
