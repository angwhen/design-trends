import requests
import urllib.request
import time
import json
#from bs4 import BeautifulSoup

#returns True or False depending on if article may be fashion related or not
#tries to err on side of false positives, since will filter more later
def is_style_article(article):
    global style_related_words
    if article['section_name'] == 'Style': #older articles do not have this
        return True
    article_json_str = json.dumps(article).lower()
        


my_data = [] #year, month, json as text
for year in range(1852,2020):
    for month in range(1,13):
        url ='https://api.nytimes.com/svc/archive/v1/%d/%d.json?api-key=LgzdHOqZOdouOwIfGS5Gdug3HTetQ0CQ'%(year,month)
        response = requests.get(url)
        data = json.loads(response.text)['response']['docs']
        for article in data:
            if is_style_article(article):
                article_json_str = json.dumps(article)
                my_data_curr = [year,1,article_json_str]
                my_data.append(my_data_curr)
        time.sleep(10)


style_related_words = []

