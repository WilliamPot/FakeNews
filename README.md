
COMP9417 Assignment2 Topic 1.5: FakeNewsChallenge
===

Introduction
---

This task involves two steps: 

(1).Determine whether a pair of headline and article is related or unrelated;

(2).Among those related pairs, determine the stance of the article to the headline('agree', 'discuss' or 'disagree').

For details of the task, see report or [FakeNewsChallenge.org](http://fakenewschallenge.org)

Files Description
---

### `fnc-1`

This folder contains original dataset and output result of our prediction model.

For details of the dataset, please see `fnc-1/README.md`.
 
### `predict.py`

This file allows us:

(1).To generate data to be tested into the folder `input/level1` and `input/level2`, then restore two pre-trained Multilayers Neural Network models (`model/trained_models/stage1/nn/model.ckpt`,`model/trained_models/stage2/nn/model.ckpt`) for prediction;

(2).The result of prediction will be saved in `fnc-1/predict.csv`;

(3).For viewing our final score, please cd to `/fnc-1` and enter `python scorer.py target.csv predict.csv` in command line.

### `Count_vectorizer.pk, tfidf_art`

These two files contain a countvectorizer model and a TF-IDF model for transforming text data into vectors.

### `utils`

This folder contains files that allow us get data for from original data set and do some validation.

### `model`

This folder contains all core files of our project including several python scripts and csv files containing transformed data.

`s1_train.py` and `s1_test.py` generate training and testing data from original dataset for step 1.

`s2_train.py` and `s2_test.py` generate training and testing data from original dataset for step 2.

`s1_nn_train.py` and `s2_nn_test.py` train the corresponding model for step 1 and step 2.

`model_input.py` contains class which can generate data from original dataset for final prediction.

`classfier.py` contains class which can restore pre-trained models and do prediction.

Getting Started
---

To get started, simply download the zip file from [Google Drive](https://drive.google.com/open?id=1tcpnds6jimzoT8oSxMW1YY0NntpH6OGk) and unzip all files.

### Prerequisites

Before any operation, the following is needed:

`python == 3.5`<br>
`Numpy == 1.14`<br>
`scikit-learn == 0.19.1`<br>
`pandas == 0.23.0`<br>
`gensim == 3.4.0`<br>
`nltk == 3.2.5`<br>
`tensorflow`<br>
`pickle`<br>

In fact, any version is OK if no error occurs after running the program.

### Installing

As transforming original data and writting them into csv files may take some time, they have been prepared already.

Just execute `predict.py`(may take some time to complete), then cd to `/fnc-1` and enter `python scorer.py target.csv predict.csv` in command line.

After this, the result including our final score will be printed out. 

On the bottom, three columns(MAX, NULL, TEST) will be displayed. The related score = TEST/MAX*100

if you directly clone files from this repository rather than download from [Google Drive](https://drive.google.com/open?id=1tcpnds6jimzoT8oSxMW1YY0NntpH6OGk), you have to follow the steps in 'Reproducing the Submission' to rebuild the whole project.
Reproducing the Submission
---

The whole procedure will take some time to be completed.

At first, cd to `/model` in command line.

### Step 1: Training model for 'Related or Unrelated' task

(1)Enter `python s1_train.py` to generate step 1 training data;

(2)Enter `python s1_test.py` to generate step 1 testing data;

(3)Enter `python s1_nn_train.py` to train step 1 model and view the result;

The trained model will be saved in `model/trained_models/stage1/nn/model.ckpt`.

You can check wheter the model correctly predicted each test data in `model/s1_test_output/nn`

### Step 2: Training model for 'Agree, Discuss or Disagree' task

(1)Enter `python s2_train.py` to generate step 2 training data;

(2)Enter `python s2_test.py` to generate step 2 testing data;

(3)Enter `python s2_nn_train.py` to train step 2 model and view the result;

The trained model will be saved in `model/trained_models/stage2/nn/model.ckpt`.

You can check wheter the model correctly predicted each test data in `model/s2_test_output/nn`

### Step 3: Use Two Model for Prediction and View the Result

(1)cd back to project directory, enter `python predict.py`,;

(2)cd to `/fnc-1` and enter `python scorer.py target.csv predict.csv`;

(3)After this, the result including our final score will be printed out. 

On the bottom, three columns(MAX, NULL, TEST) will be displayed. The related score = TEST/MAX*100

Team Member
---
* Chen Guo</br>
* Jiahui Lang</br>
* Tao Wan</br>
