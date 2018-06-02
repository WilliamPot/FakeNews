# -*- coding: utf-8 -*-
"""
Created on Wed May 30 18:27:10 2018

@author: Chen
"""

import re
from utils import dataset
import pandas as pds
from gensim import corpora, models, similarities 
from nltk.corpus import stopwords
import numpy as np
from sklearn.cross_validation import train_test_split
from gensim.models import LdaModel, TfidfModel,Word2Vec
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle as pkl

def transform(text,stoplist,model):
    t_text = [word for word in text.lower().split() if word not in stoplist]
    return model.infer_vector(t_text,steps = 50,alpha=0.15)
def write_data(batch_size,epoch,train_data,train_label):
    for i in range(epoch):
        print('Level2 testing file {} generated.'.format(i+1))
        X_output = train_data[i*batch_size:min((i+1)*batch_size,len(train_data))]
        y_output = train_label[i*batch_size:min((i+1)*batch_size,len(train_data))]
        x_pds = pds.DataFrame(X_output)
        y_pds = pds.DataFrame(y_output)
        x_pds.to_csv('test_s2/tfidf/test_data_{}.csv'.format(i+1))
        y_pds.to_csv('test_label_s2/tfidf/test_data_{}.csv'.format(i+1))
 #label unrelated:0,agree:1,discuss:2,disagree:3
cop = re.compile("[^a-z^A-Z^0-9]")

stage1_label = {"agree":0,"discuss":1,"disagree":2}

dataset = dataset.DataSet(name='competition_test')
stances = [stance for stance in dataset.stances if stance["Stance"]!="unrelated"]
articles = dataset.articles
articles_key = [key for key in articles.keys()]
articles[articles_key[0]]

stoplist = set(stopwords.words('english'))


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
        
with open('Count_vectorizer.pk', 'rb') as f:
    count_v1_voc= pkl.load(f)
count_v2= CountVectorizer(vocabulary=count_v1_voc,token_pattern=r'[A-Za-z]+')

texts_sta = [[cop.sub("", word) for word in document.lower().split() if word not in stoplist and len(cop.sub("", word))>0] for document in stances_doc]        
texts_art = [[cop.sub("", word) for word in document.lower().split() if word not in stoplist and len(cop.sub("", word))>0] for document in articles_doc]
texts_total = texts_art+texts_sta
dictionary = corpora.Dictionary(texts_total)

#corpus = [ dictionary.doc2bow(text) for text in texts_total ]

train_data = []
train_label = []
for i in range(len(stances)):
    headline = texts_sta[stance_index[i]]
    headline_text = " ".join(headline)
    article = texts_art[key_to_index[i]]
    article_text = articles_doc[key_to_index[i]]
    article_paras = sent_tokenize(article_text,"english")
    text_article_paras = [[cop.sub("", word) for word in document.lower().split() if word not in stoplist and len(cop.sub("", word))>0] for document in article_paras]
    new_article_text = [" ".join(art) for art in text_article_paras]    
    total = [headline]+text_article_paras
    new_total = [headline_text]+new_article_text
    article_dictionary = corpora.Dictionary(total)
    article_corpus = [ article_dictionary.doc2bow(text) for text in total ]
    art_model = TfidfModel(article_corpus)
    try:
        index = similarities.MatrixSimilarity(art_model[article_corpus])
    except ValueError:
        token1 = count_v2.fit_transform([headline_text]).toarray()[0]
        token2 = count_v2.fit_transform(new_article_text).toarray()[0]
        article_input = token2/max(len(token2[token2!=0]),1)
        headline_input = token1/max(len(token1[token1!=0]),1)
        train = np.concatenate([headline_input,article_input])
        train_data.append(train)
        label = stage1_label[stances[i]["Stance"]]
        labels = np.array([0,0,0])
        labels[label] = 1
        train_label.append(labels)
        continue
    
    head_line_sims = index[art_model[article_corpus]][0]
    rank_sims= []
    for j in range(1,len(head_line_sims)):
        rank_sims.append((j,head_line_sims[j]))
    rank_sorted = sorted(rank_sims, key=lambda rank_sims: rank_sims[1],reverse = True)
    new_article = ""
    length = 0
    for j in range(min(len(rank_sorted),5)):
        sentence = new_total[rank_sorted[j][0]]
        new_article += sentence
    token2 = count_v2.fit_transform([new_article]).toarray()[0]
    token1 = count_v2.fit_transform([headline_text]).toarray()[0]
    article_input = token2/max(len(token2[token2!=0]),1)
    headline_input = token1/max(len(token1[token1!=0]),1)
    train = np.concatenate([headline_input,article_input])
    train_data.append(train)
    label = stage1_label[stances[i]["Stance"]]
    labels = np.array([0,0,0])
    labels[label] = 1
    train_label.append(labels)
batch_size = 2000
epoch = len(train_data)//batch_size+1
write_data(batch_size,epoch,train_data,train_label)