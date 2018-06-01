# -*- coding: utf-8 -*-
"""
Created on Fri May 18 22:32:40 2018

@author: Chen
"""

import numpy as np
import pandas as pds
import tensorflow as tf

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    rand_index = np.random.choice(len(features),replace=False,size = batch_size)
# Shuffle, repeat, and batch the examples.
    rand_y = labels.loc[rand_index].values
    rand_x = features.loc[rand_index].values
    return rand_x,rand_y
    
tf.reset_default_graph()

learning_rate = 0.01
batch_size = 32

hidden_layer_nodes = 16
hidden_layer_nodes2 = 20
hidden_layer_nodes3 = 16


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
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_target,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
train_data = []
train_label = []
for i in range(25):
    X = pds.read_csv('train/diff_tfidf/train_data_{}.csv'.format(i+1))
    del X['Unnamed: 0']
    y = pds.read_csv('label/diff_tfidf/train_data_{}.csv'.format(i+1))
    del y['Unnamed: 0']
    train_data.append(X)
    train_label.append(y)
with tf.Session() as sess:
    sess.run(init)
    for k in range(10000):
        if (k+1)%5000 == 0:
            learning_rate = learning_rate/10
        for i in range(25):
            X = train_data[i]
            y = train_label[i]
            batch_data = train_input_fn(X,y,batch_size)
            rand_x = batch_data[0]
            rand_y = batch_data[1]
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y, keep_prob: 0.6}) 
        if (k+1)%500 == 0 or (k+1)<30:
            print("epoch {} finished.".format(k+1))
            for j in range(13):
                test_X = pds.read_csv('test/diff_tfidf/test_data_{}.csv'.format(j+1))
                del test_X['Unnamed: 0']
                test_y = pds.read_csv('test_label/diff_tfidf/test_data_{}.csv'.format(j+1))
                del test_y['Unnamed: 0']
                predict = pds.DataFrame(sess.run(correct_prediction, feed_dict={x_data: test_X.values, y_target: test_y.values, keep_prob: 1}))
                predict.to_csv('stage1_test_output/nn/predict_result_{}.csv'.format(j+1))
                print('Test set {} accuracy is: {}'.format(j+1,sess.run(accuracy, feed_dict={x_data: test_X.values, y_target: test_y.values, keep_prob: 1})))
    saver.save(sess, "trained_models/stage1/nn/model.ckpt")


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        