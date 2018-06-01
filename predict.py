# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:38:42 2018

@author: Chen
"""
import pandas as pds
import numpy as np
from model import model_input
from model import classifier
from utils import dataset_test
#generate input data.
#input_class = model_input.Generate_data()
#input_class.generate_level_1()
#input_class.generate_level_2()
target_file = 'fnc-1/competition_test_stances.csv'
predict_file = 'fnc-1/predict.csv'

label_dict = {0:'unrelated',1:'agree',2:'discuss',3:'disagree'}
estimator = classifier.Classifier()
estimator.predict()
estimator.predict_level2()
output = estimator.level1_result
output[output!=0] = estimator.level2_result
label = []
for i in range(len(output)):
    label.append(label_dict[int(output[i])])
unlabellled = dataset_test.DataSet()
unlabelled_stances = unlabellled.stances
for i in range(len(unlabelled_stances)):
    unlabelled_stances[i]['Stance'] = label[i]
unlabelled = pds.read_csv('fnc-1/competition_test_stances_unlabeled.csv')
unlabelled['Stance'] = np.array(label)
unlabelled.to_csv(predict_file,index=False)



