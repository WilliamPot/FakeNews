# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:57:16 2018

@author: Chen
"""

import re
from utils import dataset
import pandas as pds
#import feature_engineering
#import fnc_kfold
from gensim import corpora, similarities 
import numpy as np
#from sklearn.cross_validation import train_test_split
from gensim.models import TfidfModel
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import pickle as pkl

class Generate_data:
    def __init__(self):
        self.cop = re.compile("[^a-z^A-Z^0-9]")
        #self.stage1_label = {"unrelated":0,"agree":1,"discuss":1,"disagree":1}
        self.dataset = dataset.DataSet(name='competition_test')
        self.batch_size = 2000
        self.stoplist = set(stopwords.words('english'))
        #self.stage2_label = {"agree":0,"discuss":1,"disagree":2}
    def write_data(self,train_data,level):
        epoch = len(train_data)//self.batch_size+1
        for i in range(epoch):
            print('input {}'.format(i+1))
            X_output = train_data[i*self.batch_size:min((i+1)*self.batch_size,len(train_data))]
            x_pds = pds.DataFrame(X_output)
            x_pds.to_csv('input/'+level+'/input_{}.csv'.format(i+1))
        print('{} input generated.'.format(level))
    def generate_level_1(self):
        stances = self.dataset.stances
        articles = self.dataset.articles
        articles_key = [key for key in articles.keys()]
        articles[articles_key[0]]
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
        texts_sta = [[self.cop.sub("", word) for word in document.lower().split() if word not in self.stoplist and len(self.cop.sub("", word))>0] for document in stances_doc]        
        texts_art = [[self.cop.sub("", word) for word in document.lower().split() if word not in self.stoplist and len(self.cop.sub("", word))>0] for document in articles_doc]
        dictionary = corpora.Dictionary(texts_art)
        corpus = [ dictionary.doc2bow(text) for text in texts_art ]  
        
        model = TfidfModel.load('tfidf_art')
        train_data = []
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
        self.write_data(train_data = train_data,level='level1')
    def generate_level_2(self):
        stances = self.dataset.stances
        articles = self.dataset.articles
        articles_key = [key for key in articles.keys()]
        articles[articles_key[0]]  
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
        
        texts_sta = [[self.cop.sub("", word) for word in document.lower().split() if word not in self.stoplist and len(self.cop.sub("", word))>0] for document in stances_doc]        
        #texts_art = [[self.cop.sub("", word) for word in document.lower().split() if word not in self.stoplist and len(self.cop.sub("", word))>0] for document in articles_doc]
        train_data = []
        for i in range(len(stances)):
            headline = texts_sta[stance_index[i]]
            headline_text = " ".join(headline)
            #article = texts_art[key_to_index[i]]
            article_text = articles_doc[key_to_index[i]]
            article_paras = sent_tokenize(article_text,"english")
            text_article_paras = [[self.cop.sub("", word) for word in document.lower().split() if word not in self.stoplist and len(self.cop.sub("", word))>0] for document in article_paras]
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
                continue
            
            head_line_sims = index[art_model[article_corpus]][0]
            rank_sims= []
            for j in range(1,len(head_line_sims)):
                rank_sims.append((j,head_line_sims[j]))
            rank_sorted = sorted(rank_sims, key=lambda rank_sims: rank_sims[1],reverse = True)
            new_article = ""
            for j in range(min(len(rank_sorted),5)):
                sentence = new_total[rank_sorted[j][0]]
                new_article += sentence
            token2 = count_v2.fit_transform([new_article]).toarray()[0]
            token1 = count_v2.fit_transform([headline_text]).toarray()[0]
            article_input = token2/max(len(token2[token2!=0]),1)
            headline_input = token1/max(len(token1[token1!=0]),1)
            train = np.concatenate([headline_input,article_input])
            train_data.append(train)
        self.write_data(train_data = train_data,level='level2')

                 
        