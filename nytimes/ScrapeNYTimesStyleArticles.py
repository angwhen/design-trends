import requests
import urllib.request
import time
import json
import pickle
import nltk

def get_article_relevant_str(article):
    article_relevant_str =""
    if article['lead_paragraph'] != None:
        article_relevant_str +=  article['lead_paragraph']
    if article['snippet'] != None:
        article_relevant_str +=  article['snippet']
    if article['abstract'] != None:
        article_relevant_str +=  article['abstract']
    if article['headline'] != None:
        article_relevant_str +=   article['headline']['main']
    for keyword in article["keywords"]:
        article_relevant_str = article_relevant_str + keyword["value"]+ " "
    article_relevant_str = article_relevant_str.lower()
    return article_relevant_str

#returns keywords that may indicate being style related
def is_style_article(article):
    global style_related_words_unigram, style_related_words
    reasons = []
    if article['section_name'] == 'Style': #older articles do not have this
        reasons.append("Style")

    article_relevant_str = get_article_relevant_str(article)

    article_tokens = nltk.word_tokenize(article_relevant_str)
    for tok in article_tokens:
        if tok in style_related_words_unigram:
            reasons.append(tok)
    for word in style_related_words:
        if word.lower() in article_relevant_str:
            reasons.append(word.lower())
    return reasons

def get_nytimes_style_data_from_api():
    api_key = open("nytimes_api_key.txt").read().strip()
    my_data = [] #year, month, json as text
    for year in range(1852,1853):#2020):
        for month in range(1,2): #13
            url ='https://api.nytimes.com/svc/archive/v1/%d/%d.json?api-key=%s'%(year,month,api_key)
            response = requests.get(url)
            data = json.loads(response.text)['response']['docs']
            for article in data:
                style_article_reasons = is_style_article(article)
                if len(style_article_reasons) > 1:
                    article_json_str = json.dumps(article)
                    my_data_curr = [year,1,article_json_str,style_article_reasons]
                    my_data.append(my_data_curr)
            time.sleep(10)
    pickle.dump(my_data,open("data/nytimes_style_articles_data.p","wb"))

def get_style_related_words():
    style_related_words = set([])
    style_related_words_unigram = set([])

    style_related_words_unigram_temp = pickle.load(open("data/zalora_fashion_term_unigrams.p","rb"))
    for w in style_related_words_unigram_temp:
        style_related_words_unigram.add(w.lower())

    style_related_words = pickle.load(open("data/speak_fashion_fashion_terms.p","rb"))

    clothing_types_file = open("data/clothing_types_list.txt", "r")
    fabrics_file = open("data/fabrics_list.txt", "r")
    first = True
    for l in clothing_types_file:
        if not first:
            w = l.strip()
            if " " in w:
                style_related_words.add(w)
            else:
                style_related_words_unigram.add(w)
        first = False
    first = True
    for l in fabrics_file:
        if not first:
            w = l.strip()
            if " " in w:
                style_related_words.add(w)
            else:
                style_related_words_unigram.add(w)
        first = False
    return style_related_words_unigram, style_related_words

style_related_words_unigram, style_related_words = get_style_related_words()
get_nytimes_style_data_from_api()
