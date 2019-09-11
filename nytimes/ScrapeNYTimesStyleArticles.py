import requests
import urllib.request
import time
import json
import pickle
import nltk

#returns True or False depending on if article may be fashion related or not
#tries to err on side of false positives, since will filter more later
def is_style_article(article):
    global style_related_words_unigram, style_related_words
    if article['section_name'] == 'Style': #older articles do not have this
        return True

    article_json_str = json.dumps(article).lower()
    article_tokens = nltk.word_tokenize(article_json_str)
    for tok in article_tokens:
        if tok in style_related_words_unigram:
            return True
    for word in style_related_words:
        if word.lower() in article_json_str:
            return True
    return False

def get_nytimes_style_data_from_api():
    api_key = open("nytimes_api_key.txt").read().strip()
    my_data = [] #year, month, json as text
    for year in range(1852,1853):#2020):
        for month in range(1,2): #13
            url ='https://api.nytimes.com/svc/archive/v1/%d/%d.json?api-key=%s'%(year,month,api_key)
            response = requests.get(url)
            data = json.loads(response.text)['response']['docs']
            for article in data:
                if is_style_article(article):
                    article_json_str = json.dumps(article)
                    my_data_curr = [year,1,article_json_str]
                    my_data.append(my_data_curr)
            time.sleep(10)


def get_style_related_words():
    style_related_words_uni_1 = pickle.load(open("data/speak_fashion_fashion_term_unigrams.p","rb"))
    style_related_words_uni_2 = pickle.load(open("data/zalora_fashion_term_unigrams.p","rb"))
    style_related_words_unigram_temp = style_related_words_uni_1.union(style_related_words_uni_2)
    style_related_words_unigram = set([])
    for w in style_related_words_unigram_temp:
        style_related_words_unigram.add(w.lower())

    style_related_words = set([])
    clothing_types_file = open(“data/clothing_types_list.txt”, “r”)
    fabrics_file = open(“data/fabrics_list.txt”, “r”)
    first = True
    for l in clothing_types_file:
        if not first:
            style_related_words_3.add(l)
        first = False
    first = True
    for l in fabrics_file:
        if not first:
            style_related_words_3.add(l)
        first = False
    return style_related_words_unigram, style_related_words

style_related_words_unigram, style_related_words = get_style_related_words()
get_nytimes_style_data_from_api()
