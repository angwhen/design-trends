import requests
import urllib.request
import time
import json
import re
import pickle
from wordfreq import word_frequency

def get_zalora_fashion_terms():
    r = requests.get('https://www.zalora.com.hk/fashion-glossary/')
    poss_terms = re.findall(r"<b>[\w\s,.\(\)]+</b>",r.text)
    fashion_unigrams = set([])
    fashion_terms = set([])
    for el in poss_terms:
        el = el[3:-4]
        fashion_terms.add(el)
        el = el.split()
        for t in el:
            # remove overly common terms (ie "and")
            if word_frequency(t, 'en') < 0.0001:
                fashion_unigrams.add(t)
    pickle.dump(fashion_unigrams, open( "data/zalora_fashion_term_unigrams.p", "wb" ) )
    pickle.dump(fashion_terms, open( "data/zalora_fashion_terms.p", "wb" ) )


def get_speak_fashion_fashion_terms():
    fashion_unigrams = set([])
    possible_terms = []
    for letter_ord in range(97,97+26):
        letter = chr(letter_ord)
        r = requests.get('http://www.speak-fashion.de/fashion-dictionary?c2dtab=%s'%letter)
        possible_terms.extend(re.findall(r"title=\"[\w\s,.\(\)]*\"",r.text))


    # some 'terms' are repeated on every page
    # they are not actually terms
    # so i will filter them out
    possible_terms_counts = {}
    for el in possible_terms:
        if el not in possible_terms_counts:
            possible_terms_counts[el] =1
        else:
            possible_terms_counts[el] +=1

    possible_terms = set([])
    for term in possible_terms_counts:
        if possible_terms_counts[term] < 26:
            possible_terms.add(term)

    # some bad terms (too short or too common) are removed
    for el in possible_terms:
        curr_term = el.split("=")[-1][1:-1]
        curr_term = re.sub(",", " ", curr_term)
        fashion_unigrams.add(curr_term)

    pickle.dump(fashion_unigrams, open( "data/speak_fashion_fashion_terms.p", "wb" ) )

get_zalora_fashion_terms()
get_speak_fashion_fashion_terms()
