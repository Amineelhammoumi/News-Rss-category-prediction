import hashlib
from bs4 import BeautifulSoup
import requests

def getHashedString(val):
    return hashlib.md5(val.encode('utf-8')).hexdigest() 


def getArticleFromUrl(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    
    result = []
    soup = soup.find("body")
    for l in soup.find_all("p"):
        result.append(l.getText())

    return ' '.join(result)