import pandas as pd
import pickle
import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt
import math
from collections import Counter

DATA_PATH = ""
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

def make_cooccurence_matrix():
    df = pd.read_csv("%s/data/nytimes_style_articles/curated_tokenaged_parsed_only_articles_df.csv"%DATA_PATH)
    fashion_terms_occurrences = df[["curated_matched_keywords"]].apply(list).values.tolist()

    style_related_words_list = []
    for r in fashion_terms_occurrences:
        terms = r[0].replace("'",'').strip('][').split(', ')
        style_related_words_list.extend(terms)
    style_related_words_list = list(set(style_related_words_list))
    style_words_indexer = {}
    for i in range(0,len(style_related_words_list)):
        style_words_indexer[style_related_words_list[i]] = i

    mat = np.zeros([len(style_related_words_list),len(style_related_words_list)])
    for r in fashion_terms_occurrences:
        terms = r[0].replace("'",'').strip('][').split(', ')
        for el1 in terms:
            if el1 not in style_words_indexer:
                continue
            for el2 in terms:
                if el2 not in style_words_indexer:
                    continue
                mat[style_words_indexer[el1]][style_words_indexer[el2]] +=1
    pickle.dump([style_related_words_list,mat],open("%s/data/nytimes_style_articles/curated_style_related_words_cooccurence_matrix.p"%DATA_PATH,"wb"))
    return mat

def visualize_matrix():
    d = pickle.load(open("%s/data/nytimes_style_articles/curated_style_related_words_cooccurence_matrix.p"%DATA_PATH,"rb"))
    labels = d[0]
    mat = d[1]
    G = nx.from_numpy_matrix(mat)
    G = nx.relabel_nodes(G,{v: k for v, k in enumerate(labels)})
    deg_centrality_of_words = nx.degree_centrality(G)
    print (deg_centrality_of_words)
    nx.draw_spring(G)
    plt.show()

def save_deg_and_weighted_deg_centrality():
    d = pickle.load(open("%s/data/nytimes_style_articles/curated_style_related_words_cooccurence_matrix.p"%DATA_PATH,"rb"))
    df = pd.read_csv("%s/data/nytimes_style_articles/curated_tokenaged_parsed_only_articles_df.csv"%DATA_PATH)
    fashion_terms_occurrences = df[["curated_matched_keywords"]].apply(list).values.tolist()
    style_related_words_list = []
    for r in fashion_terms_occurrences:
        terms = r[0].replace("'",'').strip('][').split(', ')
        style_related_words_list.extend(terms)
    occurences_dict = Counter(style_related_words_list)

    labels = d[0]
    mat = d[1]
    term_to_deg_dict = {}
    term_to_normalized_deg_dict = {} # weighted deg is the "weighted degree sum" divided by the total occurrences
    for i,term in enumerate(labels):
        curr_deg = 0
        weighted_deg_sum = 0
        for c in mat[i]:
            weighted_deg_sum += c
            if c != 0:
                curr_deg +=1
        term_to_deg_dict[term] = curr_deg
        total_occurrences = occurences_dict[term]
        term_to_normalized_deg_dict[term] = weighted_deg_sum/total_occurrences

    G = nx.from_numpy_matrix(mat)
    G = nx.relabel_nodes(G,{v: k for v, k in enumerate(labels)})
    term_to_deg_dict = nx.degree_centrality(G)
    term_to_eig_centrality_dict = nx.eigenvector_centrality(G)

    pickle.dump(term_to_deg_dict,open("%s/data/my_data/nytimes_term_to_deg_dict.p"%FASHION_DATA_PATH,"wb"))
    pickle.dump(term_to_eig_centrality_dict,open("%s/data/my_data/nytimes_term_to_eig_centrality_dict.p"%FASHION_DATA_PATH,"wb"))
    pickle.dump(term_to_normalized_deg_dict,open("%s/data/my_data/nytimes_term_to_normalized_deg_dict.p"%FASHION_DATA_PATH,"wb"))

def make_react_code_for_graph():
    d = pickle.load(open("%s/data/nytimes_style_articles/curated_style_related_words_cooccurence_matrix.p"%DATA_PATH,"rb"))
    labels = d[0]
    #style_words_indexer = {v:i for i,v in enumerate(labels)}
    mat = d[1]
    RAND_SEL = 10
    my_str = "  this.state = { data: {\n"
    my_str +=  "nodes: ["
    for i in range(0,len(labels)):
        if i % RAND_SEL == 0:
            l = labels[i]
            my_str += "{id:\"%s\",size:%d},"%(l,max(50,min(5000,math.sqrt(sum(mat[i]))*5)))
    my_str = my_str[:-1]+ "],\n"
    my_str += "links:["

    for i in range(0,len(mat)):
        for j in range(i,len(mat)): #don't duplicate, undirected graph
            if i % RAND_SEL == 0 and j % RAND_SEL == 0 and mat[i][j] >= 1:
                my_str += "{ source: \"%s\",target: \"%s\" },"%(labels[i],labels[j])
    my_str = my_str[:-1]+"],\n},  "
    print (my_str)
    # nodes: [{ id: "Harry" }, { id: "Sally" }, { id: "Alice" }],
    #links: [{ source: "Harry", target: "Sally" }, { source: "Harry", target: "Alice" }],
    text_file = open("%s/data/react-codes/react_fashion_terms_graph.txt"%DATA_PATH, "w")
    text_file.write(my_str)
    text_file.close()

def make_react_dictionary_for_what_words_others_cooccur_with_most(top=5):
    d = pickle.load(open("%s/data/nytimes_style_articles/curated_style_related_words_cooccurence_matrix.p"%DATA_PATH,"rb"))
    labels = d[0]
    #style_words_indexer = {v:i for i,v in enumerate(labels)}
    mat = d[1]
    my_str = "{\n"
    for i in range(0,len(mat)):
        my_str += "\"%s\":["%labels[i]
        most_common_ids = np.argpartition(mat[i], -(top+1))[-(top+1):]
        if len(most_common_ids) != 1:
            for j in most_common_ids:
                if j == i:
                    continue
                my_str += "\"%s\", "%labels[j]
            my_str = my_str[:-2] + "], "
        else:
            my_str+="], "
    my_str = my_str[:-2] + "}"
    text_file = open("%s/data/react-codes/react_nytimes_fashion_terms_most_common_cooccurs.txt"%DATA_PATH, "w")
    text_file.write(my_str)
    text_file.close()

def make_react_dictionary_for_what_adjs_other_cooccur_with_most(top=100):
    import nltk
    df = pd.read_csv("%s/data/nytimes_style_articles/curated_tokenaged_parsed_only_articles_df.csv"%DATA_PATH)
    fashion_terms_occurrences= df[["curated_matched_keywords"]].apply(list).values.tolist()
    texts_in_same_order = df[["main_parts_text"]].apply(list).values.tolist()

    style_related_words_to_adjs_dict = {}
    for i in range(0,len(fashion_terms_occurrences)):
        r = fashion_terms_occurrences[i]
        terms = r[0].replace("'",'').strip('][').split(', ')
        print (terms)
        for curr_term in terms:
            print (curr_term)
            # FIND RELATED ADJS
            curr_adjs_before = []
            pos_tagged = nltk.pos_tag(nltk.word_tokenize(texts_in_same_order[i]))
            my_inds = []
            for j in range(pos_tagged):
                if pos_tagged[0].lower() == curr_term:
                    my_inds.append(j)
            for j in my_inds:
                curr_adj_ind_tester = j-1
                while curr_adj_ind_tester >= 0 and pos_tagged[curr_adj_ind_tester][1] in ['JJ','JJR','JJS']:
                    curr_adjs_before.append( pos_tagged[curr_adj_ind_tester].lower())
                    curr_adj_ind_tester-1
            print (curr_adjs_before)
            # ADD RELATED ADJS TO DICT
            if curr_term not in  style_related_words_to_adjs_dict:
                style_related_words_to_adjs_dict[curr_term] = {}
            for adj in curr_adjs_before:
                if adj not in style_related_words_to_adjs_dict[curr_term]:
                    style_related_words_to_adjs_dict[curr_term][adj] = 0
                style_related_words_to_adjs_dict[curr_term][adj] += 1
    pickle.dump(style_related_words_to_adjs_dict ,open("%s/data/nytimes_style_articles/curated_style_words_to_adjectives_dict.p"%DATA_PATH,"wb"))

    my_str = "{\n"
    for term in fashion_terms_occurrences:
        my_str += "\"%s\":["%term
        related_adjs = sorted([(adjs,count) for k in style_related_words_to_adjs_dict[term].keys()],key=lambda x: x[1])
        if len(related_adjs) != 0:
            for adj,cnt in related_adjs:
                my_str += "\"%s\", "%adj
            my_str = my_str[:-2] + "], "
        else:
            my_str+="], "
    my_str = my_str[:-2] + "}"
    text_file = open("%s/data/react-codes/react_nytimes_fashion_terms_related_adjs.txt"%DATA_PATH, "w")
    text_file.write(my_str)
    text_file.close()


#make_cooccurence_matrix()
#visualize_matrix()
#save_deg_and_weighted_deg_centrality()
#make_react_code_for_graph()
#make_react_dictionary_for_what_words_others_cooccur_with_most()
make_react_dictionary_for_what_adjs_other_cooccur_with_most()
