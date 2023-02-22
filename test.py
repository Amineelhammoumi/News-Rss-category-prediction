

import numpy as np

import spacy

# Chargement de spacy configuré à partir des ressources de langue française
nlp = spacy.load("en_core_web_sm")


import nltk


from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from xml.dom import minidom
stop_words = stopwords.words("english")
def read(fname):
    # convert an XML file (fname) into a list of dicts
    data = []

    en_db = minidom.parse(fname)
    items = en_db.getElementsByTagName('item')
    for i in range(len(items)):
        txt=items[i].getElementsByTagName('title')[0].childNodes[0].data +" "+items[i].getElementsByTagName('description')[0].childNodes[0].data +  " " +  items[i].getElementsByTagName('text')[0].childNodes[0].data
        
   
        data.append(txt)
    return data 


def gety(fname):
    # convert an XML file (fname) into a list of dicts
    data = []

    en_db = minidom.parse(fname)
    items = en_db.getElementsByTagName('item')
    for i in range(len(items)):
        txt=items[i].getElementsByTagName('category')[0].childNodes[0].data
        
   
        data.append(txt)
    return data 

y  = gety('data2.xml')






data  = read('data2.xml')
target = ['art', 'economy', 'sports', 'politics', 'medical', 'science'] 



nombre_de_mots = list(map(lambda x: len(x.split(" ")), data))

nombre_de_car = list(map(lambda x: len(x), data))


def avg_word(sentence):
 words = sentence.split()
 return (sum(len(word) for word in words)/len(words))

longueurs_moyennes_mots = list(map(lambda x: avg_word(x), data))
print("longueurs moyennes des mots dans le corpus: ",
np.mean(longueurs_moyennes_mots))




nombre_mots_vides = list(map(lambda x: len([x for x in x.split() if x in
stop_words]), data))



nombre_de_numériques = list(map(lambda x: len([x for x in x.split()
if x.isdigit()]), data))





data=list(map(lambda x: " ".join(x.lower() for x in x.split()),data))


tokenized_sent_spacy = []
for i in range(len(data)):
    sent = nlp(data[i])
    tokenized_sent_spacy.append([X.text for X in sent])
       




import re
filtered_sent = []
for i in range(len(tokenized_sent_spacy)):
    #Suppression de la ponctuation
    tokenized_sent_spacy[i]=list(map(lambda x: re.sub('[^\w\s]','',x),tokenized_sent_spacy[i]))
    #Suppression des chaines de caractéres vide dans la liste ex: ['Recently', '', 'National']
    tokenized_sent_spacy[i]=list((filter(None, tokenized_sent_spacy[i])))
    


 
# les mots "vides"
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))





for i in range(len(tokenized_sent_spacy)):
    text = {}
    text = tokenized_sent_spacy[i]
    temp = []
    for w in text:
        if w not in stop_words:
            temp.append(w)
    #print(temp)
    filtered_sent.append(temp)
       




from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english')

stemmed_words = []
for i in range(len(filtered_sent)):
    temp = []
    for w in filtered_sent[i]:
        temp.append(stemmer.stem(w))
    stemmed_words.append(temp)



train_data = []
for i in range(len(stemmed_words)):
    sent = " ".join(stemmed_words[i])
    train_data.append(sent)
#print(train_data)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
v = TfidfVectorizer()
X = []
y=[]
X = read("data2.xml")
X = np.array(X)
y = np.array(y)
y=['politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'politics', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'economy', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'art', 'medical', 'medical', 'medical', 'medical', 'medical', 'medical', 'medical', 'medical', 'medical', 'medical', 'science', 'science', 'science', 'science', 'science', 'science', 'science', 'science', 'science', 'science', 'science', 'science', 'science', 'science', 'science', 'science', 'science', 'science', 'science', 'science']



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)



cv = TfidfVectorizer()
X_train = cv.fit_transform(X_train).toarray()
X_test = cv.fit_transform(X_test).toarray()


clf_knn = KNeighborsClassifier(n_neighbors = 5)
clf_knn.fit(X_train, y_train)

y_pred_knn = clf_knn.predict(X_test)


print("Accuracy knn: ", metrics.accuracy_score(y_test, y_pred_knn))



clf_nb = MultinomialNB()
clf_nb.fit(X_train, y_train)
y_pred_nb = clf_nb.predict(X_test)
print("Accuracy MultinomialNB: ", metrics.accuracy_score(y_test, y_pred_nb))



from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

import json


resultat = read('benchmark_en.xml')


X_prof=resultat
v.fit(X_prof)
testX = v.transform(X_prof)

pred_prof = clf_knn.predict(testX)
pred_prof = pred_prof.tolist()

print(pred_prof)

probs=clf_knn.predict_proba(testX)
probs = probs.tolist()

print(probs)
lang = 'en' 

res=dict()

res['pred']=list(pred_prof) # list of predicted labels
res['probs']=list(probs)
