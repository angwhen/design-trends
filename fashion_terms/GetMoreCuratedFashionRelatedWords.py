import requests
import urllib.request
import time
import json
import pickle
from nltk.tag import pos_tag
import pandas as pd

def make_style_words_df():
    style_related_words = set([])
    style_related_words_unigram = set([])

    style_related_words_speak_fashion = pickle.load(open("../nytimes/data/speak_fashion_fashion_terms.p","rb"))
    style_related_words_zalora = pickle.load(open("data/zalora_fashion_terms.p","rb"))
    for w in style_related_words_speak_fashion.union(style_related_words_zalora):
        if " " in w:
            style_related_words.add(w.strip())
        else:
            style_related_words_unigram.add(w.strip())

    clothing_types_file = open("../nytimes/data/clothing_types_list.txt", "r")
    fabrics_file = open("../nytimes/data/fabrics_list.txt", "r")
    first = True
    for l in clothing_types_file:
        if not first:
            w = l.lower().strip()
            if " " in w:
                style_related_words.add(w)
            else:
                style_related_words_unigram.add(w)
        first = False
    first = True
    for l in fabrics_file:
        if not first:
            w = l.lower().strip()
            if " " in w:
                style_related_words.add(w)
            else:
                style_related_words_unigram.add(w)
        first = False

    all_style_related_words = list(style_related_words_unigram.union(style_related_words))

    all_words = []
    for word in style_related_words_unigram:
        p = pos_tag(word.split())
        all_words.append([word,1,p,1])
    for word in style_related_words:
        p = pos_tag(word.split())
        all_words.append([word,0,p,1]) # setting all human edited labels to 1, unless otherwise edited

    # columns of the dataframe should be: WORD, Unigram or not, POS tagged
    df=pd.DataFrame(all_words)
    df.columns = ["word","unigram","pos_tags","human_edited_label"]
    df.to_csv("data/my_data/fashion_terms.csv")
