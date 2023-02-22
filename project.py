import pandas as pd
import numpy as np
import json
import text_preprocessing as tp

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn import metrics

df = pd.read_json("data.json")

X = []
y = []

for index, row in df.iterrows():
    content = tp.prepare(row['title'] +" "+ row['text'] +" "+ row['description'])
    X.append(content)
    y.append(row['category'])

X = np.array(X , dtype=object)
y = np.array(y , dtype=object)



print(X)

le = preprocessing.LabelEncoder()
"""

X = le.fit_transform(X)

X = X.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)

clf_knn = KNeighborsClassifier(n_neighbors = 5)
clf_knn.fit(X_train, y_train)

y_pred_knn = clf_knn.predict(X_test)


print("Accuracy: ", metrics.accuracy_score(y_test, y_pred_knn))


from xml.dom import minidom 

counter = 0
data = []
en_db = minidom.parse("benchmark_en.xml") 

items = en_db.getElementsByTagName('item') 

predictions = []
probabilities = []

for i in range(len(items)): 
    counter += 1
    title = items[i].getElementsByTagName('title')[0].childNodes[0].data 
    desc = items[i].getElementsByTagName('description')[0].childNodes[0].data 
    text = items[i].getElementsByTagName('text')[0].childNodes[0].data 
    
    data.append(title + " " + desc + " " + text)

    if counter == 10:
        X_prof = data
        
        label_encoder = preprocessing.LabelEncoder()
        X_prof = label_encoder.fit_transform(X_prof)
        X_prof = X_prof.reshape(-1,1)

        pred_prof = clf_knn.predict(X_prof)
        predictions += pred_prof.tolist()

        probs = clf_knn.predict_proba(X_prof)
        probabilities += probs.tolist()
    
        data = []
        counter = 0


with open("result_predictions.json", "w") as o:
    json.dump(predictions, o)

with open("result_probabilities.json", "w") as o:
    json.dump(probabilities, o)
    """