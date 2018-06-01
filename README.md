
COMP9417 Assignment2 Topic 1.5: FakeNewsChallenge
===
1.Introduction
---
This task involves two steps: 

(1).Determine whether a number of pairs of headline and article is related or unrelated;

(2).Among those related pairs, determine the stance of the article to the headline('agree', 'discuss' or 'disagree').

For details of the task, see report or [FakeNewsChallenge.org](http://fakenewschallenge.org)

2.Files Description
---
### `fnc-1`

This folder contains original dataset and output result from our prediction model.

For details of the dataset, please see `fnc-1/README.md`
 
### `predict.py`

This file allows user:

(1).To generate input data into the folder `input/level1` and `input/level2`, then restore two pre-trained Multilayers Neural Network models (`/model/trained_models/stage1/nn/model.ckpt`,`/model/trained_models/stage2/nn/model.ckpt`) for prediction;

(2).The result of prediction will be saved in `fnc-1/predict.csv`;

(3).For viewing our final score, please cd to `/fnc-1` and enter `scorer.py target.csv predict.csv` in command line.

### `Count_vectorizer.pk, tfidf_art`

These two files contain a countvectorizer model and a TF-IDF model for transforming text data into vectors.

### `utils`

This folder contains files that allow user get data for training and testing from original data set.

3.Getting Started
---
