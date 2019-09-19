import pandas as pd
import json
import pickle
import nltk

def snippet(row):
    j =  json.loads(row.unparsed_text)
    if "snippet" in j:
        return j["snippet"]
    return None

def return_common_prevs(year,month,my_word="dress"):
    global df
    # import pickle for the year
    # find all the articles in the month
    # send the article to json
    # get the snippet
    # find all the words occuring before "dress"0
    curr_df = df[(df.year == year) & (df.month == month) & (df.unparsed_text.str.contains('dress'))][["unparsed_text"]]
    curr_df["snippet"] = curr_df.apply(snippet, axis = 1)
    snippets = curr_df["snippet"].values.tolist()
    word_dict = {}
    for snippet in snippets:
        tokens = nltk.word_tokenize(snippet.lower())
        tokens = [el for el in tokens if el.isalpha()]
        if my_word in tokens:
            i = tokens.index(my_word)
            if i != 0:
                prev_word = tokens[i-1]
                if prev_word in word_dict:
                    word_dict[prev_word] +=1
                else:
                    word_dict[prev_word] = 1

    return word_dict

df = pd.read_csv("data/nytimes_style_articles/unparsed_articles_df.csv")
"""
#cannot use this dataframe rn, because it is taking forever to populate .... maybe can't load such a big one at once?
df = pd.read_csv("data/nytimes_style_articles/parsed_only_articles_df.csv")
df2  = df[df.matched_keywords.apply(lambda x: 'dress' in x)][["year","month","snippet","lead_paragraph","abstract"]]
df2 = df2.dropna(thresh=3)
print (df2.head(50))
"""

print (return_common_prevs(2000,1))
