# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:03:28 2018

@author: Chen
"""
import numpy as np
import pandas as pds
import tensorflow as tf

class Classifier:
    def __init__(self):
        self.stage1_label = {0:"unrelated",1:"related"}
        self.level1_result = np.array([])
        self.level2_result = np.array([])
        self.level2_index = []
    def predict(self):
        tf.reset_default_graph()
        hidden_layer_nodes = 16
        hidden_layer_nodes2 = 20        
        
        keep_prob = tf.placeholder(tf.float32)  
        x_data= tf.placeholder(shape = [None,10],dtype = tf.float32,name='x_input')
        y_target = tf.placeholder(shape = [None,2],dtype = tf.float32,name='y_input')
        A1 = tf.Variable(tf.truncated_normal(shape=[10,hidden_layer_nodes]))
        tf.summary.histogram('weights1',A1)
        b1 = tf.Variable(tf.truncated_normal(shape=[hidden_layer_nodes]))
        tf.summary.histogram('bias1',b1)
        hidden_output = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(x_data,A1),b1)),keep_prob)
        
        A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,hidden_layer_nodes2]))
        tf.summary.histogram('weights2',A2)
        b2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes2]))
        tf.summary.histogram('bias2',b2)
        hidden_output2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(hidden_output,A2),b2)),keep_prob)
             
        A3 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes2,2]))
        tf.summary.histogram('weights3',A3)
        b3 = tf.Variable(tf.random_normal(shape=[2]))
        tf.summary.histogram('bias3',b3)
        y_predict = tf.add(tf.matmul(hidden_output2,A3),b3)
        
        cross_entropy = tf.reduce_mean(
           tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y_predict))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        
        correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_target,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,"model/trained_models/stage1/nn/model.ckpt")
            #sess.run(init)
            for j in range(13):
                test_X = pds.read_csv('input/level1/input_{}.csv'.format(j+1))
                del test_X['Unnamed: 0']
                X = test_X.values
                predict = sess.run(tf.argmax(y_predict,1), feed_dict={x_data: X,keep_prob: 1})
                self.level1_result = np.concatenate([self.level1_result,predict])
                level2_input_index = predict==1
                self.level2_index.append(level2_input_index)
                #predict.to_csv('stage1_test_output/nn/predict_result_{}.csv'.format(j+1))
                #print('Test set {} result is: {}'.format(j+1,sess.run(accuracy, feed_dict={x_data: test_X.values, y_target: test_y.values, keep_prob: 1})))       
    def predict_level2(self):
        tf.reset_default_graph()
        hidden_layer_nodes = 80
        hidden_layer_nodes2 = 100      
        
        keep_prob = tf.placeholder(tf.float32)  
        x_data= tf.placeholder(shape = [None,10000],dtype = tf.float32,name='x_input')
        y_target = tf.placeholder(shape = [None,3],dtype = tf.float32,name='y_input')
        A1 = tf.Variable(tf.truncated_normal(shape=[10000,hidden_layer_nodes]))
        tf.summary.histogram('weights1',A1)
        b1 = tf.Variable(tf.truncated_normal(shape=[hidden_layer_nodes]))
        tf.summary.histogram('bias1',b1)
        hidden_output = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(x_data,A1),b1)),keep_prob)
        
        A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,hidden_layer_nodes2]))
        tf.summary.histogram('weights2',A2)
        b2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes2]))
        tf.summary.histogram('bias2',b2)
        hidden_output2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(hidden_output,A2),b2)),keep_prob)
             
        A3 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes2,3]))
        tf.summary.histogram('weights3',A3)
        b3 = tf.Variable(tf.random_normal(shape=[3]))
        tf.summary.histogram('bias3',b3)
        y_predict = tf.add(tf.matmul(hidden_output2,A3),b3)
        
        cross_entropy = tf.reduce_mean(
           tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y_predict))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        
        correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_target,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,"model/trained_models/stage2/nn/model.ckpt")
            for j in range(13):
                test_X = pds.read_csv('input/level2/input_{}.csv'.format(j+1))
                del test_X['Unnamed: 0']
                X = test_X.values[self.level2_index[j]]
                predict = sess.run(tf.argmax(y_predict,1), feed_dict={x_data: X,keep_prob: 1})
                self.level2_result = np.concatenate([self.level2_result,predict+1])
                #predict.to_csv('stage1_test_output/nn/predict_result_{}.csv'.format(j+1))
                #print('Test set {} result is: {}'.format(j+1,sess.run(accuracy, feed_dict={x_data: test_X.values, y_target: test_y.values, keep_prob: 1})))       