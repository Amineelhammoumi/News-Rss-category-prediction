#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import feedparser

from urllib.error import HTTPError
from urllib.error import URLError
from xml.dom.minidom import parseString
from dicttoxml import dicttoxml


#----------------
# get uri content
#----------------
import requests
from bs4 import BeautifulSoup

# Création d'une instance
news_feed = feedparser.parse('http://rss.cnn.com/rss/cnn_allpolitics.rss')
science_feed = feedparser.parse('http://feeds.feedburner.com/AllDiscovermagazinecomContent')
sport_feed = feedparser.parse("https://api.foxsports.com/v1/rss?partnerKey=zBaFxRyGKCfxBagJG9b8pqLyndmvo7UU")
health_feed = feedparser.parse('http://rssfeeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC')
art_feed = feedparser.parse("https://www.thisiscolossal.com/feed/")
economic_rss = feedparser.parse("https://www.ft.com/global-economy?format=rss")

news_feed.entries




'''

# Propriétés du flux
print("news keys: " ,news_feed.feed.keys())
print("science keys :",science_feed.feed.keys())
print("sport_keys : ",sport_feed.feed.keys())
print("health_keys : ",health_feed.feed.keys())
print("art_keys : ",art_feed.feed.keys())
print("economic_keys :" , economic_rss.feed.keys())




for entry in news_feed.entries:
    print(f"{entry.title} --> {entry.link}")
    
    
for entry in science_feed.entries:
    print(f"{entry.title} --> {entry.link}")
    
    
for entry in sport_feed.entries:
    print(f"{entry.title} --> {entry.link}")

for entry in health_feed.entries:
    print(f"{entry.title} --> {entry.link}")

for entry in art_feed.entries:
    print(f"{entry.title} --> {entry.link}")
    

for entry in economic_rss.entries:
    print(f"{entry.title} --> {entry.link}")
    
    
'''
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

texts = []
dicp = {}

"""
if __name__ == "__main__":
    for entry in news_feed.entries:
        s = entry.link
        i = getURL(s)
        texts.append(i)


"""
#case1 =  {"title" : news_feed.entries[0].title , "description" : news_feed.entries[0].description , "text" : getURL(news_feed.entries[0].link).replace("\n","")}

#print(case#1)



politis_list = []

for entry in news_feed.entries:
    case = {"item" : {"title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  }
    politis_list.append(case)
    
xml = dicttoxml(politis_list ,  attr_type = False)
dom = parseString(xml)
print(dom.toprettyxml())
xml_decode = xml.decode()
xmlfile = open("dict.xml", "w" ,  encoding="utf-8")
xmlfile.write(xml_decode)
xmlfile.close()
