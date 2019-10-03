import pandas as pd
import pickle
import numpy as np
from collections import Counter

df = pd.read_csv("data/nytimes_style_articles/unparsed_articles_df.csv")
date_and_fashion_terms = df[["year","month","matched_keywords"]]
year_keywords_df = df.groupby('year')['matched_keywords'].apply(list).reset_index()
month_keywords_df = df.groupby(['year','month'])['matched_keywords'].apply(list).reset_index()
year_keywords_list = year_keywords_df[['year',"matched_keywords"]].values.tolist()
year_keywords_list.sort(key=lambda x: x[0])
#month_keywords_list = month_keywords_df[['month',"matched_keywords"]].values.tolist()
#month_keywords_list.sort(key=lambda x: x[0])

flatten = lambda l: [item for sublist in l for item in sublist]

all_words_list = []
for row in year_keywords_list:
    keywords = flatten([el[2:-2].split("', '" ) for el in row[1]])
    all_words_list.extend(keywords)
total_word_counts_dict = Counter(all_words_list)

my_str = "yearly_fashion_terms:["
for row in year_keywords_list:
    year = row[0]
    keywords = flatten([el[2:-2].split("', '" ) for el in row[1] if el != "Style"])
    cnts = [[tup[0],tup[1]/total_word_counts_dict[tup[0]]] for tup in Counter(keywords).most_common(50)]
    my_str += "[%s, %s],\n"%(year,cnts)
my_str = my_str[:-2]+"],\n"
"""
my_str += "monthly_fashion_terms:["
for row in month_keywords_list:
    month = row[0]
    keywords = flatten([el[2:-2].split("', '" ) for el in row[1] if el != "Style"])
    cnts = [[tup[0],tup[1]/total_word_counts_dict[tup[0]]] for tup in Counter(keywords).most_common(50)]
    my_str += "[%s, %s],\n"%(month,cnts)
my_str = my_str[:-2]+"],\n"
"""

text_file = open("data/react_key_fashion_terms_by_date.txt", "w")
text_file.write(my_str)
text_file.close()