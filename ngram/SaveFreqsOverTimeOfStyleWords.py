import pickle
import sys
import time
# using this getngrams: https://github.com/econpy/google-ngrams/blob/master/getngrams.py
sys.path.insert(1, './google-ngrams')
import getngrams

style_related_words = pickle.load(open("../data/style_related_words_unigram_list.p","rb"))

i = 0
while i < len(style_related_words):
    curr_arg_str = ""
    orig_i = i
    for j in range(0,5):
        print (i)
        if i == len(style_related_words):
            break
        curr_arg_str += style_related_words[i] +","
        i+=1
    curr_arg_str = curr_arg_str[:-1]
    curr_arg_str += " --startYear=1800 --endYear=2008 -caseInsensitive"
    res = getngrams.runQuery(curr_arg_str)
    if len(res) == 0:
        i = orig_i # try again, maybe timed out
        time.sleep(60)
    time.sleep(6)
