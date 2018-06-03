# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:43:26 2018

@author: Chen
"""
import re
from utils import dataset
import pandas as pds
from gensim import corpora
from nltk.corpus import stopwords
import numpy as np
from gensim.models import TfidfModel


def transform(text,stoplist,model):
    t_text = [word for word in text.lower().split() if word not in stoplist]
    return model.infer_vector(t_text,steps = 50,alpha=0.15)
 #label unrelated:0,agree:1,discuss:2,disagree:3
cop = re.compile("[^a-z^A-Z^0-9]")

stage1_label = {"unrelated":0,"agree":1,"discuss":1,"disagree":1}

dataset = dataset.DataSet()
stances = dataset.stances
articles = dataset.articles
articles_key = [key for key in articles.keys()]
articles[articles_key[0]]
#list_stopWords=list(set(stopwords.words('english')))
stoplist = set(stopwords.words('english'))
#punctuation = string.punctuation+'‘’‛“”„‟'
#for key in articles_key:
    #for punc in punctuation:
        #articles[key] = articles[key].replace(punc,'')
    #articles[key] = cop.sub("", articles[key])
#for stance in stances:
    #for punc in punctuation:
        #stance["Headline"] = stance["Headline"].replace(punc,'')
    #stance["Headline"] = cop.sub("", stance["Headline"])
stances_set = set([stance["Headline"] for stance in stances])
stances_doc = [stance for stance in stances_set]
keys = [key for key in articles.keys()]
key_to_index=  []
stance_index = []
for stance in stances:
    for i in range(len(stances_doc)):
        if stance["Headline"] == stances_doc[i]:
            stance_index.append(i)
            break
    for j in range(len(keys)):
        if stance["Body ID"] == keys[j]:
            key_to_index.append(j)
            break


articles_doc = [articles[key] for key in articles.keys()]
texts_sta = [[cop.sub("", word) for word in document.lower().split() if word not in stoplist and len(cop.sub("", word))>0] for document in stances_doc]        
texts_art = [[cop.sub("", word) for word in document.lower().split() if word not in stoplist and len(cop.sub("", word))>0] for document in articles_doc]
dictionary = corpora.Dictionary(texts_art)
corpus = [ dictionary.doc2bow(text) for text in texts_art ]  

model = TfidfModel.load('tfidf_art')
train_data = []
train_label = []
for i in range(len(stances)):
    stance = texts_sta[stance_index[i]]
    art_corpu = corpus[key_to_index[i]]
    tdidf_corpu = model[art_corpu]
    tfidf_sorted = sorted(tdidf_corpu, key=lambda tdidf_corpu: tdidf_corpu[1],reverse = True)
    
    pos_output = [0]*10#diff
    neg_output = [0]*10
    pos_index = 0
    neg_index = 0
    for j in range(len(tfidf_sorted)):
        word = dictionary.get(tfidf_sorted[j][0]) 
        tfidf_value = tfidf_sorted[j][1]           
        if word in stance:
            if pos_index!=10:
                pos_output[pos_index] = tfidf_value
                pos_index += 1   
        else:
            if neg_index!=10:
                neg_output[neg_index] = -tfidf_value
                neg_index += 1   
        if neg_index == 10 and pos_index == 10:
            break    
    train_data.append(np.add(pos_output,neg_output))
    
    label = stage1_label[stances[i]["Stance"]]
    #train_data.append(np.concatenate((stance,article),axis=0))
    labels = np.array([0,0])
    labels[label] = 1
    train_label.append(labels)
batch_size = 2000
epoch = len(train_data)//batch_size+1
rand_index = np.random.choice(len(train_data),replace=False,size = len(train_data))
train_data = np.array(train_data)[rand_index]
train_label = np.array(train_label)[rand_index]
def write_data(train_data,train_label):
    for i in range(epoch):
        print('Level1 training file {} generated.'.format(i+1))
        X_output = train_data[i*batch_size:min((i+1)*batch_size,len(train_data))]
        y_output = train_label[i*batch_size:min((i+1)*batch_size,len(train_data))]
        x_pds = pds.DataFrame(X_output)
        y_pds = pds.DataFrame(y_output)
        x_pds.to_csv('train/diff_tfidf/train_data_{}.csv'.format(i+1))
        y_pds.to_csv('label/diff_tfidf/train_data_{}.csv'.format(i+1))