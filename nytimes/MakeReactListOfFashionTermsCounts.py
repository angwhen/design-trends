import pandas as pd
import pickle
import numpy as np
from collections import Counter

DATA_PATH = "."
try:
    f=open("data_location.txt", "r")
    DATA_PATH  = f.read().strip()
except:
    print ("data is right here")

df = pd.read_csv("%s/data/nytimes_style_articles/curated_tokenaged_parsed_only_articles_df.csv"%DATA_PATH)
date_and_fashion_terms = df[["year","month","curated_matched_keywords"]]
year_keywords_df = df.groupby('year')['curated_matched_keywords'].apply(list).reset_index()
month_keywords_df = df.groupby(['year','month'])['curated_matched_keywords'].apply(list).reset_index()
year_keywords_list = year_keywords_df[['year',"curated_matched_keywords"]].values.tolist()
year_keywords_list.sort(key=lambda x: x[0])

flatten = lambda l: [item for sublist in l for item in sublist]

all_words_list = []
for row in year_keywords_list:
    keywords = flatten([el[2:-2].split("', '" ) for el in row[1]])
    all_words_list.extend(keywords)
total_word_counts_dict = Counter(all_words_list)

year_to_fashion_terms_list_dict = {}
my_str = "yearly_fashion_terms:["
for row in year_keywords_list:
    year = row[0]
    keywords = flatten([el[2:-2].split("', '" ) for el in row[1] if el != "Style"])
    cnts = [[tup[0],tup[1]/total_word_counts_dict[tup[0]]] for tup in Counter(keywords).most_common(50)]
    my_str += "[%s, %s],\n"%(year,cnts)
    year_to_fashion_terms_list_dict[int(year)] = keywords

my_str = my_str[:-2]+"],\n"
text_file = open("%s/data/react-codes/react_key_fashion_terms_by_date.txt"%DATA_PATH, "w")
text_file.write(my_str)
text_file.close()

pickle.dump(year_to_fashion_terms_list_dict,open("%s/data/year_to_fashion_terms_list_dict.p"%DATA_PATH,"wb"))
