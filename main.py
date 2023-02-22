
# %%


import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import nltk
nltk.download('stopwords')
stop_words=set(stopwords.words("english"))
import string
from xml.dom import minidom
import json

import pandas as pd

dataset = pd.read_json("myitems.json")

print(dataset)
dataset.head()

print(dataset.shape)
print()
print(dataset.info())
print()
print(dataset['category'].value_counts())
target_category = dataset['category'].unique()
print(target_category)

dataset['categoryId'] = dataset['category'].factorize()[0]
print(dataset.head())


category = dataset[['category', 'categoryId']].drop_duplicates().sort_values('categoryId')
print(category)

def remove_tags(text):
    remove = re.compile(r'')
    return re.sub(remove, '', text)


def special_char(text):
    reviews = ''
    for x in text:
        if x.isalnum():
            reviews = reviews + x
        else:
            reviews = reviews + ' '
    return reviews


dataset['text'] = dataset['text'].apply(remove_tags)


def convert_lower(text):
    return text.lower()

dataset['text'] = dataset['text'].apply(convert_lower)
dataset['text'][1]

def remove_stopwords(text):
  stop_words = set(stopwords.words('english'))
  words = word_tokenize(text)
  return [x for x in words if x not in stop_words]


dataset['text'] = dataset['text'].apply(remove_stopwords)
dataset['text'][1]


def lemmatize_word(text):
  wordnet = WordNetLemmatizer()
  return " ".join([wordnet.lemmatize(word) for word in text])


dataset['text'] = dataset['text'].apply(lemmatize_word)
dataset['text'][1]




x = dataset['text']
y = dataset['categoryId']

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


x = np.array(dataset.iloc[:,0].values)
y = np.array(dataset.categoryId.values)

print("X.shape = ",x.shape)
print("y.shape = ",y.shape)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0, shuffle = True)

cv = TfidfVectorizer()
x_train = cv.fit_transform(x_train).toarray()
x_test = cv.fit_transform(x_test).toarray()


print(len(x_train))
print(len(x_test))


perform_list = [ ]


def run_model(model_name, est_c, est_pnlty):

    mdl=''



    if model_name == 'Random Forest':

        mdl = RandomForestClassifier(n_estimators=100 ,criterion='entropy' , random_state=0)

    elif model_name == 'Multinomial Naive Bayes':

        mdl = MultinomialNB(alpha=1.0,fit_prior=True)

    elif model_name == 'K Nearest Neighbour':

        mdl = KNeighborsClassifier(n_neighbors=10 , metric= 'minkowski' , p = 4)



    oneVsRest = OneVsRestClassifier(mdl)

    oneVsRest.fit(x_train, y_train)
    
    y_pred = oneVsRest.predict(x_test)



    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)


    precision, recall, f1score, support = score(y_test, y_pred, average='micro')

    print(f'Test Accuracy Score of Basic {model_name}: % {accuracy}')

    print(f'Precision : {precision}')

    print(f'Recall : {recall}')

    print(f'F1-score : {f1score}')



    perform_list.append(dict([

        ('Model', model_name),

        ('Test Accuracy', round(accuracy, 2)),

        ('Precision', round(precision, 2)),

        ('Recall', round(recall, 2)),

        ('F1', round(f1score, 2))

    ]))

    return oneVsRest




def getfile(fname):
    donnees = []
    en_db = minidom.parse(fname)
    it = en_db.getElementsByTagName('item')
    for i in range(len(it)):   
        filter_title=it[i].getElementsByTagName('title')[0].childNodes[0].data.lower()
        filter_title = filter_title.translate(str.maketrans('', '', string.punctuation))
        
        tokenized_title=word_tokenize(filter_title)
        filtered_sent=[]
        for w in tokenized_title:
            if w not in stop_words:
                filtered_sent.append(w)
        filter_title = ' '.join(filtered_sent)

        filter_description=it[i].getElementsByTagName('description')[0].childNodes[0].data.lower()
        filter_description = filter_description.translate(str.maketrans('', '', string.punctuation))
        
        tokenized_desc=word_tokenize(filter_description)
        filtered_sent=[]
        for w in tokenized_desc:
            if w not in stop_words:
                filtered_sent.append(w)
        filter_description = ' '.join(filtered_sent)
        
        filter_text=it[i].getElementsByTagName('text')[0].childNodes[0].data.lower()
        filter_text = filter_text.translate(str.maketrans('', '', string.punctuation))
        
        tokenized_txt=word_tokenize(filter_text)
        filtered_sent=[]
        for w in tokenized_txt:
            if w not in stop_words:
                filtered_sent.append(w)
        filter_text = ' '.join(filtered_sent)
        sent = filter_title + '' + filter_description + '' + filter_text
        donnees.append(sent)  
    return donnees


def getCategory(yy):
    result = ""
    if yy == [0]:
        result = "SCIENCE"
    elif yy == [1]:
        result = "ECONOMY"
    elif yy == [2]:
        result = "SPORT"
    elif yy == [3]:
        result = "HEALTH"
    elif yy == [4]:
        result = "ART_CULTURE"
    elif yy == [5]:
        result = "POLITIQUE"

    return result

mdl = run_model('Multinomial Naive Bayes', est_c=None, est_pnlty=None)



bench = getfile("benchmark_en.xml")
data = []

y_pred1 = cv.transform(bench).toarray()
pred = mdl.predict(y_pred1).tolist()
prob = mdl.predict_proba(y_pred1).tolist()

output = { "pred": pred, "prob": prob}

with open("Amine_ElHammoumi_KNN_EN.res", "w") as out:
    json.dump(output, out)



