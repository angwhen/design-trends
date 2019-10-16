import spacy
import pickle
nlp = spacy.load('en_core_web_md')

style_related_words_list = pickle.load(open("../data/style_related_words_unigram_list.p","rb"))
style_words_to_vec_dict = {}
for w in style_related_words_list:
    doc = nlp(w)
    style_words_to_vec_dict[w] = doc[0].vector

print (style_words_to_vec_dict)

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
