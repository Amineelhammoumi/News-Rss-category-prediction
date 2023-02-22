import spacy

## LOAD SPACY WITH ENGLISH RESOURCES
nlp = spacy.load("en_core_web_sm")

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


def split_into_sentences(item):
    return sent_tokenize(item)


def split_into_words(items):
    res = []
    for s in items:
        words_list = [X.text for X in nlp(s)]
        for w in words_list:
            res.append(w)

    return res    


def get_stop_words():
    return stopwords.words("english")


def filter_empty_wordd(words):
    filtered_sent = []
    for w in words:
        if w not in get_stop_words():
            filtered_sent.append(w)
    
    return filtered_sent



def apply_stem(words):
    stemmer = SnowballStemmer(language='english')
    stemmed_words=[stemmer.stem(X) for X in words]

    return stemmed_words


def vectorize(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    X = X.reshape(-1, 1)
    X= [X]
    return X
    
    

def prepare(text):
    output = split_into_sentences(text)
    # output = split_into_words(output)
    # output = filter_empty_wordd(output)
    # output = apply_stem(output)

    return output
    # return vectorize(output)