import pandas as pd
import pickle
import numpy as np
from collections import Counter
import os

style_related_words = pickle.load(open("../data/style_related_words_unigram_list.p","rb"))

years_to_terms_to_counts_dict = {}
terms_sums_dict = {} #for tfidf
directory = os.fsencode("./data/ngram_data")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        df = pd.read_csv("./data/ngram_data/%s"%filename)
        df = df.set_index('year')
        curr_dict = df.to_dict(orient='index')
        for y in curr_dict:
            if y not in years_to_terms_to_counts_dict:
                years_to_terms_to_counts_dict[y] = curr_dict[y]
            else:
                for term in curr_dict[y]:
                    years_to_terms_to_counts_dict[y][term] = curr_dict[y][term]
            for term in curr_dict[y]:
                if term in terms_sums_dict:
                    terms_sums_dict[term] += curr_dict[y][term]
                else:
                    terms_sums_dict[term] = curr_dict[y][term]
    else:
        continue

my_str = "yearly_google_ngram_fashion_terms:["
for year in range(1800,2008):
    cnts = [[term,years_to_terms_to_counts_dict[year][term]/terms_sums_dict[term]] for term in style_related_words if term in years_to_terms_to_counts_dict[year]]
    my_str += "[%s, %s],\n"%(year,cnts)
my_str = my_str[:-2]+"],\n"
text_file = open("data/react_key_fashion_terms_by_date_from_google_ngram.txt", "w")
text_file.write(my_str)
text_file.close()
