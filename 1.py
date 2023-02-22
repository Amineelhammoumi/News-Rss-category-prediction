import feedparser

import numpy as np
from urllib.error import HTTPError
from urllib.error import URLError
from xml.dom.minidom import parseString
from nltk.corpus import stopwords
from dicttoxml import dicttoxml

from nltk.tokenize import word_tokenize

import json

import requests
from bs4 import BeautifulSoup
import re

politics_feed = feedparser.parse("https://thepoliticalinsider.com/feed/")

economy_feed= feedparser.parse("https://johnhcochrane.blogspot.com/feeds/posts/default")

art_feed= feedparser.parse("http://feeds.hyperallergic.com/hyperallergic")

medical_feed= feedparser.parse("https://www.news-medical.net/syndication.axd?format=rss")

sports_feed= feedparser.parse("https://www.foxsports.com/rss-feeds")

science_feed= feedparser.parse("http://feeds.feedburner.com/sciencealert-latestnews")


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
    case = {"category" : "politics" ,"title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)
    
    
    

for entry in economy_feed.entries:
    case = {"category" : "economy","title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)  



for entry in art_feed.entries:
    case = {"category" : "art","title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)  
 
    
for entry in medical_feed.entries:
    case = {"category" : "medical","title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)  
    
        

for entry in sports_feed.entries:
    case = {"category" : "sports","title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)  
 
    
for entry in science_feed.entries:
    case = {"category" : "science","title" : entry.title , "description" : entry.description , "text" : getURL(entry.link).replace("\n","")}  
    feed_list.append(case)  
    
xml = dicttoxml(feed_list ,  attr_type = False)
dom = parseString(xml)
print(dom.toprettyxml())
xml_decode = xml.decode()
xmlfile = open("data1.xml", "w" ,  encoding="utf-8")
xmlfile.write(xml_decode)
xmlfile.close()
    