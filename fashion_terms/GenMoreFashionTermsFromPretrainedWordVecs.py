import spacy
import pickle
import numpy as np
from scipy.spatial import distance
nlp = spacy.load('en_core_web_md')

def idk():
    style_related_words_list = pickle.load(open("../data/style_related_words_unigram_list.p","rb"))
    style_words_to_vec_dict = {}
    for w in style_related_words_list:
        doc = nlp(w)
        style_words_to_vec_dict[w] = doc[0].vector

    print (style_words_to_vec_dict)


def get_distance_from_center_of_unigram_terms_in_nytimes_style_dataset():
    d = pickle.load(open("../nytimes/data/nytimes_style_articles/style_related_words_cooccurence_matrix.p","rb"))
    words = d[0]
    style_words_to_vec_dict = {}
    for w in words:
        doc = nlp(w)
        style_words_to_vec_dict[w] = doc.vector

    vecs_list = np.array(list(style_words_to_vec_dict.values()))
    mean_vec = np.mean(vecs_list,axis=0)

    style_words_to_dist_from_center_dict = {}
    for w in style_words_to_vec_dict:
        style_words_to_dist_from_center_dict[w] = distance.euclidean(style_words_to_vec_dict[w],mean_vec)

    pickle.dump(style_words_to_dist_from_center_dict,open("data/my_data/terms_to_distance_from_vector_center_dict.p","wb"))

get_distance_from_center_of_unigram_terms_in_nytimes_style_dataset()
"""import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data
# https://fasttext.cc/docs/en/english-vectors.html
d = load_vectors('wiki-news-300d-1M.vec')
print (d)
"""
