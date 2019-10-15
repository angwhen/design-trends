import pandas as pd
import pickle
import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt
import math

def make_cooccurence_matrix():
    df = pd.read_csv("data/nytimes_style_articles/unparsed_articles_df.csv")
    #style_related_words_list = pickle.load(open("../data/style_related_words_unigram_list.p","rb"))

    fashion_terms_occurrences = df[["matched_keywords"]].apply(list).values.tolist()

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
    pickle.dump([style_related_words_list,mat],open("data/nytimes_style_articles/style_related_words_cooccurence_matrix.p","wb"))
    return mat

def visualize_matrix():
    d = pickle.load(open("data/nytimes_style_articles/style_related_words_cooccurence_matrix.p","rb"))
    labels = d[0]
    mat = d[1]
    G = nx.from_numpy_matrix(mat)
    G = nx.relabel_nodes(G,{v: k for v, k in enumerate(labels)})
    deg_centrality_of_words = nx.degree_centrality(G)
    nx.draw_spring(G)
    plt.show()

def make_react_code_for_graph():
    d = pickle.load(open("data/nytimes_style_articles/style_related_words_cooccurence_matrix.p","rb"))
    labels = d[0]
    #style_words_indexer = {v:i for i,v in enumerate(labels)}
    mat = d[1]
    RAND_SEL = 20
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
    my_str = my_str[:-1]+"],\n}}"
    print (my_str)
    # nodes: [{ id: "Harry" }, { id: "Sally" }, { id: "Alice" }],
    #links: [{ source: "Harry", target: "Sally" }, { source: "Harry", target: "Alice" }],
    text_file = open("data/react_fashion_terms_graph.txt", "w")
    text_file.write(my_str)
    text_file.close()



#make_cooccurence_matrix()
#visualize_matrix()
make_react_code_for_graph()
