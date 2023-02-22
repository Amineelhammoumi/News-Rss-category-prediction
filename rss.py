import feedparser
import hashlib
import warnings
import json
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

warnings.filterwarnings("ignore")

feeders = {
    'SCIENCE': [
        'http://feeds.feedburner.com/sciencealert-latestnews'
    ],
    'ECONOMY': [
        'https://johnhcochrane.blogspot.com/feeds/posts/default'
    ],
    'SPORT': [
        'https://www.sportingnews.com/us/rss'
    ],
    'HEALTH': [
        'http://feeds.feedburner.com/Medgadget'
    ],
    'ART_CULTURE': [
        'http://feeds.hyperallergic.com/hyperallergic'
    ],
    'POLITIQUE': [
        "https://thepoliticalinsider.com/feed/"
    ]
}

result = []
for feed in feeders:
    for link in feeders[feed]:
        res = feedparser.parse(link)

        for item in res['entries']:
            row = {
                'category': feed,
                'title': item.title,
                'description': item.description,
                'text': getArticleFromUrl(item.link)
            }
            result.append(row)

result = json.dumps(result, indent=4)
with open('myitems.json', 'w') as outfile:
    outfile.write(result)