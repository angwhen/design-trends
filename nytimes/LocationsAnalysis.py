import pandas as pd
import pickle
import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt
import math
from collections import Counter

DATA_PATH = "."
try:
    f=open("data_location.txt", "r")
    DATA_PATH  = f.read().strip()
except:
    print ("data is right here")

FASHION_DATA_PATH = "."
try:
    f=open("fashion_data_location.txt", "r")
    FASHION_DATA_PATH  = f.read().strip()
except:
    print ("data is right here")

# make map from years to locations
# make react word cloud for this
def make_map_from_years_to_locations():
    df = pd.read_csv("%s/data/nytimes_style_articles/locationed_curated_tokenaged_parsed_only_articles_df.csv"%DATA_PATH)
    year_keywords_df = df.groupby('year')['cities'].apply(list).reset_index()
    year_keywords_list = year_keywords_df[['year',"cities"]].values.tolist()
    year_keywords_list.sort(key=lambda x: x[0])

    flatten = lambda l: [item for sublist in l for item in sublist]

    """all_words_list = [] # in case want to do tfidf
    for row in year_keywords_list:
        keywords = flatten([el[2:-2].split("', '" ) for el in row[1]])
        all_words_list.extend(keywords)
    total_word_counts_dict = Counter(all_words_list)"""

    year_to_cities_list_dict = {}
    my_str = "["
    for row in year_keywords_list:
        year = row[0]
        keywords = flatten([el[2:-2].split("', '" ) for el in row[1]])
        cnts = [[tup[0],tup[1]] for tup in Counter(keywords).most_common(50) if tup[0] != ""]
        my_str += "[%s, %s],\n"%(year,cnts)
        year_to_cities_list_dict[int(year)] = keywords

    my_str = my_str[:-2]+"]\n"
    text_file = open("%s/data/react-codes/react_locations_by_date.txt"%DATA_PATH, "w")
    text_file.write(my_str)
    text_file.close()

    pickle.dump(year_to_cities_list_dict,open("%s/data/year_to_cities_list_dict.p"%DATA_PATH,"wb"))


# make map from terms to locations
# add to table
def make_map_from_terms_to_locations():
    df = pd.read_csv("%s/data/nytimes_style_articles/locationed_curated_tokenaged_parsed_only_articles_df.csv"%DATA_PATH)
    keywords_locations_list = df[["curated_matched_keywords","cities"]].values.tolist()
    flatten = lambda l: [item for sublist in l for item in sublist]

    terms_to_locations_dict = {}
    for row in keywords_locations_list:
        keywords = flatten([el[2:-2].split("', '" ) for el in row[0]])
        cities = flatten([[city for city in el[2:-2].split("', '" ) if city != ""] for el in row[1]])
        print (keywords)
        print (cities)
        for term in keywords:
            if term not in terms_to_locations_dict:
                terms_to_locations_dict[term] = []
            terms_to_locations_dict[term].extend(cities)
    pickle.dump(terms_to_locations_dict,open("%s/data/terms_to_cities_dict.p"%DATA_PATH,"wb"))

    my_str = "{\n"
    for term in terms_to_locations_dict:
        locations = terms_to_locations_dict[term]
        locations_counter = Counter(locations).most_common(8)
        locations_counter.sort(key=lambda x: x[1])
        my_str += "\"%s\":["%term
        most_common = [tup[0] for tup in locations_counter if tup[0] != ""]
        if len(most_common) == 0:
            my_str += "], "
        else:
            for loc in most_common:
                my_str += "\"%s\", "%loc
            my_str = my_str[:-2] + "], "

    my_str = my_str[:-2] + "}"
    text_file = open("%s/data/react-codes/react_terms_to_most_common_cities.txt"%DATA_PATH, "w")
    text_file.write(my_str)
    text_file.close()
    print ("Done")

#make_map_from_years_to_locations()
make_map_from_terms_to_locations()
