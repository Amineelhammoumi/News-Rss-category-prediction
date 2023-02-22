#!/usr/bin/env python
# coding: utf-8

# # Prétraitements

# In[ ]:


## Installer spacy
#> pip3 install --user spacy
## Charger les ressources pour le Français
#> python3 -m spacy download fr_core_news_sm

## Installer nltk
#> pip3 install --user nltk


# In[2]:



import spacy
import nltk
nltk.download('punkt')
nltk.download('stopwords')
# Chargement de spacy configuré à partir des ressources de langue française
nlp = spacy.load("fr_core_news_sm")
from nltk.tokenize import sent_tokenize

# Découpage en phrase
text="""Une nouvelle semaine décisive s'ouvre lundi dans la lutte contre le coronavirus. Alors que le bilan 
s'est de nouveau alourdi lundi en France, avec 418 décès supplémentaires, les autorités espèrent (enfin) observer 
les premiers impacts du confinement. Suivez l'évolution de la situation en direct."""

tokenized_text=sent_tokenize(text)
print(tokenized_text)


# In[3]:


from nltk.tokenize import word_tokenize

# Découpage en mot de la 1ere phrase
tokenized_sent=word_tokenize(tokenized_text[0])
print(tokenized_sent)


# In[4]:


# Un meilleur Tokenizer avec spacy
sent = nlp(tokenized_text[0])
tokenized_sent_spacy = [X.text for X in sent]
print(tokenized_sent_spacy)


# In[5]:


from nltk.corpus import stopwords
stop_words=set(stopwords.words("french"))
print(stop_words)


# In[6]:


# Suppression des mots "vides"
filtered_sent=[]
for w in tokenized_sent_spacy:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_sent_spacy)
print("Filterd Sentence:",filtered_sent)


# In[7]:


# Stemming
# Le stemming est un processus de normalisation linguistique, qui réduit les mots à leur racine ou coupe les affixes 
# dérivés. Par exemple, le mot "connexion", "connecté", "connecter" se réduit à un mot commun "connect".
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='french')

stemmed_words=[stemmer.stem(X) for X in tokenized_sent_spacy]

print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)


# In[8]:


# Reconnaissance d'entité nommées (NER)
text="""L’Organisation mondiale de la santé sur le pied de guerre : l’OMS a appelé tous les pays du globe à 
accentuer les programmes de dépistage des populations au coronavirus. Le 18 mars, elle a exhorté l’Afrique, encore 
peu touchée, à « se préparer au pire ».
"""
tokenized_text=sent_tokenize(text)
print(tokenized_text)


# In[9]:


# Reconnaissance d'entité nommées (NER)
sent = nlp(tokenized_text[0])
entites=[(X.text, X.label_) for X in sent.ents]
print(entites)


# In[10]:


sent = nlp(tokenized_text[1])
entites=[(X.text, X.label_) for X in sent.ents]
print(entites)


# # SKLearn Vectorizer

# In[11]:


# Installing SKLearn
#> pip3 install --user scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ["Une nouvelle semaine décisive s'ouvre lundi dans la lutte contre le coronavirus.", 
          "Alors que le bilan s'est de nouveau alourdi lundi en France, avec 418 décès supplémentaires, \
          les autorités espèrent (enfin) observer les premiers impacts du confinement.", 
          "Suivez l'évolution de la situation en direct."]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
vocabulary = vectorizer.get_feature_names_out()
print(vocabulary)
print("Nombre de mots : ", len(vocabulary))


# In[12]:


# Affiche le vecteur associé à la première phrase (1er document)
print(X[0])


# In[14]:


print(vocabulary[8], vocabulary[22], vocabulary[7], vocabulary[25], vocabulary[21], "...")


# In[15]:


# Affiche le vecteur associé à la deuxième phrase (2eme document)
print(X[1])


# In[16]:


# Affiche le type de la matrice X

print(type(X))


# In[17]:


# Affiche le contenu de X
print("X:")
print(X)


# In[ ]:




