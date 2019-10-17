import pickle

nytimes_term_to_deg_dict = pickle.load(open("data/my_data/nytimes_term_to_deg_dict.p","rb"))
nytimes_term_to_normalized_deg_dict = pickle.load(open("data/my_data/nytimes_term_to_normalized_deg_dict.p","rb"))
nytimes_term_to_eig_centrality_dict = pickle.load(open("data/my_data/nytimes_term_to_eig_centrality_dict.p","rb"))

all_terms = list(nytimes_term_to_deg_dict.keys())
all_terms.extend( list(nytimes_term_to_normalized_deg_dict.keys()))
all_terms.extend( list(nytimes_term_to_eig_centrality_dict.keys()))
all_terms = list(set(all_terms))
my_str = " ["
#term: "Raymond",
#score1: 10,
#score2: 20
#}];"
for t in all_terms:
    my_str += "{term:\"%s\","%t
    if t in nytimes_term_to_deg_dict:
        my_str += "score1: %f, "%nytimes_term_to_deg_dict[t]
    if t in nytimes_term_to_normalized_deg_dict:
        my_str += "score2: %f, "%nytimes_term_to_normalized_deg_dict[t]
    if t in nytimes_term_to_eig_centrality_dict:
        my_str += "score3: %f, "%nytimes_term_to_eig_centrality_dict[t]
    my_str = my_str[:-2] + "},\n"
my_str = my_str[:-2] + "];"

text_file = open("data/my_data/react_fashion_terms_fashionability_scores.txt", "w")
text_file.write(my_str)
text_file.close()
