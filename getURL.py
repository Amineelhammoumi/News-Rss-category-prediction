import nltk
from urllib.error import HTTPError
from urllib.error import URLError


#----------------
# get uri content
#----------------
import requests
from bs4 import BeautifulSoup
def getURL(link):
    print(link)
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
	    print(text)
    return text


if __name__ == "__main__":
    link = 'https://www.ft.com/content/5c1140e8-5cd5-4b74-8ee6-4b70f22ca6ba'
    s = getURL(link)
    print(s)
