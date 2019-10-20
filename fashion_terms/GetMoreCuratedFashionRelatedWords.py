import requests
import urllib.request
import time
import json
import pickle
from nltk.tag import pos_tag
import pandas as pd
import csv


def make_style_words_df():
    style_related_words = set([])
    style_related_words_unigram = set([])

    style_related_words_speak_fashion = pickle.load(open("../nytimes/data/speak_fashion_fashion_terms.p","rb"))
    style_related_words_zalora = pickle.load(open("../nytimes/data/zalora_fashion_terms.p","rb"))
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

    human_labels_dict = {}
    with open('data/my_data/human_labelled_fashion_terms.csv', 'rb') as f:
        reader = csv.reader(f)
        human_labels = list(reader)
    for w in human_labels:
        if len(w[1]) != 0:
            human_labels_dict[w[0]] = int(w[1])

    all_words = []
    for word in style_related_words_unigram:
        p = pos_tag(word.split())

        label = 1
        if "homepage" in word.lower() or ".com" in word.lower():
            label = 0
        if word in human_labels_dict:
            label = human_labels_dict[word]

        proper = 0
        if len(p) >=2:
            all_proper = True
            for el in p:
                if el[1] != "NNP":
                    all_proper = False
            if all_proper:
                proper = 1

        all_words.append([word,1,label,proper,p])
    for word in style_related_words:
        p = pos_tag(word.split())

        label = 1
        if "homepage" in word.lower() or ".com" in word.lower():
            label = 0
        if word in human_labels_dict:
            label = human_labels_dict[word]

        proper = 0
        if len(p) >=2:
            all_proper = True
            for el in p:
                if el[1] != "NNP":
                    all_proper = False
            if all_proper:
                proper = 1

        all_words.append([word,0,label,proper,p]) # default is 1 unless other reason

    # columns of the dataframe should be: WORD, Unigram or not, POS tagged, my label
    # my label is 0 for do not use, 1 for use, 2 for depends on the context
    df=pd.DataFrame(all_words)
    df.columns = ["word","unigram","human_edited_label","proper","pos_tags"]
    df.to_csv("data/my_data/fashion_terms.csv")

make_style_words_df()
