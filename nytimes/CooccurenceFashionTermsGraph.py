import pandas as pd
import pickle
import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt

def make_cooccurence_matrix():
    df = pd.read_csv("data/nytimes_style_articles/unparsed_articles_df.csv")
    style_related_words_list = pickle.load(open("../data/style_related_words_unigram_list.p","rb"))
    #print (style_related_words_list)
    style_words_indexer = {}
    for i in range(0,len(style_related_words_list)):
        style_words_indexer[style_related_words_list[i]] = i

    mat = np.zeros([len(style_related_words_list),len(style_related_words_list)])
    fashion_terms_occurrences = df[["matched_keywords"]].apply(list).values.tolist()


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

    my_str = "data: {\n"
    my_str +=  "nodes: ["
    for l in labels:
        my_str += "{id:\"%s\"}, "%l
    my_str = my_str[:-1]+ "],\n"
    my_str += "links:["

    for i in range(0,len(mat)):
        for j in range(i+1,len(mat)): #don't duplicate, undirected graph
            if mat[i][j] != 0:
                my_str += "{ source: \"%s\", target: \"%s\" }, "%(labels[i],labels[j])
    my_str = my_str[:-1]+"],\n}"

    # nodes: [{ id: "Harry" }, { id: "Sally" }, { id: "Alice" }],
    #links: [{ source: "Harry", target: "Sally" }, { source: "Harry", target: "Alice" }],
    text_file = open("data/react_fashion_terms_graph.txt", "w")
    text_file.write(my_str)
    text_file.close()



#make_cooccurence_matrix()
#visualize_matrix()
make_react_code_for_graph()
