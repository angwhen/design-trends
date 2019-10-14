import pandas as pd
import pickle
import numpy as np
import json

def make_cooccurence_matrix():
    df = pd.read_csv("data/nytimes_style_articles/unparsed_articles_df.csv")
    style_related_words_list = pickle.load(open("../data/style_related_words_unigram_list.p","rb"))
    style_words_indexer = {}
    for i in range(0,len(style_related_words_list)):
        style_words_indexer[style_related_words_list[i]] = i

    mat = np.zeros([len(style_related_words_list),len(style_related_words_list)])
    fashion_terms_occurrences = df[["matched_keywords"]]
    for r in fashion_terms_occurrences:
        for el1 in r[0]:
            for el2 in r[0]:
                mat[style_words_indexer[el1]][style_words_indexer[el2]] +=1
    pickle.dump([style_related_words_list,mat],open("data/nytimes_style_articles/style_related_words_cooccurence_matrix.p","wb"))
    return mat

mat = make_cooccurence_matrix()

print (mat)
