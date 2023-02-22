# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 20:50:45 2022

@author: hp
"""

import feedparser

import numpy as np
from urllib.error import HTTPError
from urllib.error import URLError
from xml.dom.minidom import parseString
from nltk.corpus import stopwords
from dicttoxml import dicttoxml

from nltk.tokenize import word_tokenize


import requests
from bs4 import BeautifulSoup
import re

politics_feed = feedparser.parse("https://thepoliticalinsider.com/feed/")
politics2_feed= feedparser.parse("https://reason.com/feed/")
economy_feed= feedparser.parse("https://johnhcochrane.blogspot.com/feeds/posts/default")
economy2_feed= feedparser.parse("https://timharford.com/feed/")
art_feed= feedparser.parse("http://feeds.hyperallergic.com/hyperallergic")
art2_feed= feedparser.parse("https://www.artforum.com/rss.xml")
medical_feed= feedparser.parse("https://www.news-medical.net/syndication.axd?format=rss")
medical2_feed= feedparser.parse("http://feeds.feedburner.com/Medgadget")
sports_feed= feedparser.parse("https://www.foxsports.com/rss-feeds")
sports2_feed= feedparser.parse("https://www.espn.com/espn/news/story?page=rssinfo")
science_feed= feedparser.parse("http://feeds.feedburner.com/sciencealert-latestnews")
science2_feed= feedparser.parse("https://www.newscientist.com/feed/home/?cmpid=RSS%7CNSNS-Home")

def getURL(link):
   
    text=''
    try:
        response=requests.get(link)
    except HTTPError as e:
	    print('The server couldn\'t fulfill the request.')
	    print('Error code: ', e.code)
    except URLError as e:
	    print('We failed to reach a server.')
	    print('Reason: ', e.reason)
    except requests.RequestException as e:
        print('We failed to connect.')
        print('Reason: ', e)
    else:
	    # everything is fine
	    response=requests.get(link)
	    html = response.text
	    soup = BeautifulSoup(html,'lxml')
	    for tag in soup.findAll('p'):
	        tag_text=tag.getText()
	        text = text+"\n"+tag_text	     
	
    return text
feed_list = []




for entry in politics_feed.entries:
    case = {"title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)
    
    
for entry in politics2_feed.entries:
    case = {"title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)    

for entry in economy_feed.entries:
    case = {"title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)  

for entry in economy2_feed.entries:
    case = {"title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)  

for entry in art_feed.entries:
    case = {"title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)  

for entry in art2_feed.entries:
    case = {"title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)  
    
for entry in medical_feed.entries:
    case = {"title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)  
    
        
for entry in medical2_feed.entries:
    case = {"title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)  
    
for entry in sports_feed.entries:
    case = {"title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)  
    
for entry in sports2_feed.entries:
    case = {"title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)  
    
for entry in science_feed.entries:
    case = {"title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)  
    
for entry in science2_feed.entries:
    case = {"title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)  
    
"""
xml = dicttoxml(feed_list ,  attr_type = False)
dom = parseString(xml)
print(dom.toprettyxml())
xml_decode = xml.decode()
xmlfile = open("amine.xml", "w" ,  encoding="utf-8")
xmlfile.write(xml_decode)
xmlfile.close()

    """
    
    


stop_words = set(stopwords.words('english'))
taille = feed_list.size
    
wordsFiltreda=[]
for i in range(taille):
    tokenized_title=word_tokenize(feed_list[i]["title"])
    for w in tokenized_title:
        if w not in stop_words:
            wordsFiltreda.append(w)


wordsFiltredb=[]
for i in range(taille):
    tokenized_title=word_tokenize(feed_list[i]["summary"])
    data=list(map(lambda x: " ".join(x.lower() for x in x.split()),tokenized_title))
    data=list(map(lambda x: re.sub('[^\w\s]','',x),data))
    for w in data:
        if w not in stop_words:
            wordsFiltredb.append(w)

wordsFiltredc=[]
for i in range(taille):
    tokenized_title=word_tokenize(feed_list[i]["text"])
    data=list(map(lambda x: " ".join(x.lower() for x in x.split()),tokenized_title))
    data=list(map(lambda x: re.sub('[^\w\s]','',x),data))
    for w in data:
        if w not in stop_words:
            wordsFiltredc.append(w)


e = np.concatenate((wordsFiltreda,wordsFiltredb,wordsFiltredc,))



print(e)
    

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


print("Load train and test sets")
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
print('#train:', len(newsgroups_train.data))
print('#test:', len(newsgroups_test.data),"\n")
print("#Categories: ", len(newsgroups_train.target_names), ' ',newsgroups_train.target_names,"\n")
print('Category of Train doc 0:', newsgroups_train.target_names[newsgroups_train.target[0]]
      , newsgroups_train.data[0])
print('_________', newsgroups_train.target[0])
print(newsgroups_train.target)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)

print("matrix of vectors shape for train data")
print(vectors.shape)
print("type of vectors:", type(vectors[0]))
print("non zero average dimensions of the vectors: ", vectors.nnz / float(vectors.shape[0]))

print()
print("The vectors are sparse vectors with different nz dimensions")
print("#nz dimensions of vector[0]:", vectors[0].nnz)
print("#nz dimensions of vector[2]:", vectors[2].nnz)

import time
print("train the KNN classifier")
clf_knn = KNeighborsClassifier(n_neighbors=3)
start = time.time()
clf_knn.fit(vectors, newsgroups_train.target)
end = time.time()
print('elapsed time for KNN training: ', end - start, ' sec')
print()
print("predict with the KNN classifier")
start = time.time()
pred_knn = clf_knn.predict(vectors_test)
end = time.time()
print('elapsed time for KNN testing: ', end - start, ' (sec)')
print()


print("Evaluate the KNN classifier")
F1_knn=metrics.f1_score(newsgroups_test.target, pred_knn, average='macro')
PREC_knn=metrics.precision_score(newsgroups_test.target, pred_knn, average='macro')
REC_knn=metrics.recall_score(newsgroups_test.target, pred_knn, average='macro')
print("KNN F1=", F1_knn)
print("KNN PREC=", PREC_knn)
print("KNN REC=", REC_knn)


print("train the NB Multinomial classifier")
start = time.time()
clf_nb = MultinomialNB(alpha=.01)
clf_nb.fit(vectors, newsgroups_train.target)
end = time.time()
print('elapsed time for NB training: ', end - start, ' sec')
print()
print("predict with the NB classifier")
start = time.time()
pred_NB = clf_nb.predict(vectors_test)
prob_NB = clf_nb.predict_proba(vectors_test)

end = time.time()
print('elapsed time for NB testing: ', end - start, ' (sec)')
print(prob_NB)


print("evaluate the NB classifier")
F1_NB=metrics.f1_score(newsgroups_test.target, pred_NB, average='macro')
PREC_NB=metrics.precision_score(newsgroups_test.target, pred_NB, average='macro')
REC_NB=metrics.recall_score(newsgroups_test.target, pred_NB, average='macro')
print("NB F1=", F1_NB)
print("NB PREC=", PREC_NB)
print("NB REC=", REC_NB)
print(metrics.confusion_matrix(newsgroups_test.target, pred_NB))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    