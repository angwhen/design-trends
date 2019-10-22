import pickle

FASHION_DATA_PATH = "."
try:
    f=open("fashion_data_location.txt", "r")
    FASHION_DATA_PATH  = f.read().strip()
except:
    print ("data is right here")

nytimes_term_to_deg_dict = pickle.load(open("%s/data/my_data/nytimes_term_to_deg_dict.p"%FASHION_DATA_PATH,"rb"))
nytimes_term_to_normalized_deg_dict = pickle.load(open("%s/data/my_data/nytimes_term_to_normalized_deg_dict.p"%FASHION_DATA_PATH,"rb"))
nytimes_term_to_eig_centrality_dict = pickle.load(open("%s/data/my_data/nytimes_term_to_eig_centrality_dict.p"%FASHION_DATA_PATH,"rb"))
terms_to_distance_from_vector_center_dict = pickle.load(open("%s/data/my_data/terms_to_distance_from_vector_center_dict.p"%FASHION_DATA_PATH,"rb"))

all_terms = list(nytimes_term_to_deg_dict.keys())
all_terms.extend( list(nytimes_term_to_normalized_deg_dict.keys()))
all_terms.extend( list(nytimes_term_to_eig_centrality_dict.keys()))
all_terms.extend( list(terms_to_distance_from_vector_center_dict.keys()))
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
    if t in terms_to_distance_from_vector_center_dict:
        my_str += "score4: %f, "%terms_to_distance_from_vector_center_dict[t]
    my_str = my_str[:-2] + "},\n"
my_str = my_str[:-2] + "];"

text_file = open("data/react-codes/react_fashion_terms_fashionability_scores.txt", "w")
text_file.write(my_str)
text_file.close()
