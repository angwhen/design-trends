import pandas as pd
import json
from textblob import TextBlob
import nltk
import gc

DATA_PATH = ""
try:
    f=open("data_location.txt", "r")
    DATA_PATH  = f.read().strip()
except:
    print ("data is right here")

def type_of_material(row):
    j =  json.loads(row.unparsed_text)
    if "type_of_material" in j:
        return j["type_of_material"]
    return None

def web_url(row):
    j =  json.loads(row.unparsed_text)
    if "web_url" in j:
        return j["web_url"]
    return None

def headline(row):
    j =  json.loads(row.unparsed_text)
    if "headline" in j:
        return j["headline"]
    return None

def word_count(row):
    j =  json.loads(row.unparsed_text)
    if "word_count" in j:
        return j["word_count"]
    return None

def abstract(row):
    j =  json.loads(row.unparsed_text)
    if "abstract" in j:
        return j["abstract"]
    return None

def snippet(row):
    j =  json.loads(row.unparsed_text)
    if "snippet" in j:
        return j["snippet"]
    return None

def lead_paragraph(row):
    j =  json.loads(row.unparsed_text)
    if "lead_paragraph" in j:
        return j["lead_paragraph"]
    return None

def keywords(row):
    j =  json.loads(row.unparsed_text)
    if "keywords" in j:
        return j["keywords"]
    return None

def section_name(row):
    j =  json.loads(row.unparsed_text)
    if "section_name" in j:
        return j["section_name"]
    return None

def subsection_name(row):
    j =  json.loads(row.unparsed_text)
    if "subsection_name" in j:
        return j["subsection_name"]
    return None

def pub_date(row):
    j =  json.loads(row.unparsed_text)
    if "pub_date" in j:
        return j["pub_date"]
    return None

def get_main_parts(row):
    txt = ""
    if type(row.headline) == str:
        txt += row.headline + " "
    if type(row.abstract) == str:
        txt += str(row.abstract) + " "
    if type(row.snippet) == str:
        txt += row.snippet + " "
    if type(row.lead_paragraph) == str:
        txt += row.lead_paragraph + " "
    return txt

def get_noun_phrases(row):
    txt = row.main_parts_text
    blob = TextBlob(txt)
    return blob.noun_phrases

def get_nouns(row):
    txt = row.main_parts_text
    nouns = []
    for word,pos in nltk.pos_tag(nltk.word_tokenize(txt)):
         if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
             nouns.append(word)
    return nouns

def get_adjectives(row):
    txt = row.main_parts_text
    adjs = []
    for word,pos in nltk.pos_tag(nltk.word_tokenize(txt)):
         if (pos == 'JJ' or pos == 'JJR' or pos == 'JJS'):
             adjs.append(word)
    return adjs

def unparsed_to_parsed():
    print ("Starting from unparsed")
    df = pd.read_csv("%s/data/nytimes_style_articles/unparsed_articles_df.csv"%DATA_PATH)
    print ("Read unparsed")

    df["type_of_material"] = df.apply(type_of_material, axis = 1)
    df["web_url"] = df.apply(web_url, axis = 1)
    df["headline"] = df.apply(headline, axis = 1)
    df["word_count"] = df.apply(word_count, axis = 1)
    df["abstract"] = df.apply(abstract, axis = 1)
    df["snippet"] = df.apply(snippet, axis = 1)
    df["lead_paragraph"] = df.apply(lead_paragraph, axis = 1)
    df["keywords"] = df.apply(keywords, axis = 1)
    df["section_name"] = df.apply(section_name, axis = 1)
    df["subsection_name"] = df.apply(subsection_name, axis = 1)
    df["pub_date"] = df.head(5).apply(pub_date, axis = 1)

    df.to_csv("%s/data/nytimes_style_articles/parsed_articles_df.csv"%DATA_PATH)
    print ("All done parsing")

def parsed_to_parsed_without_unparsed_text():
    df = pd.read_csv("%s/data/nytimes_style_articles/parsed_articles_df.csv"%DATA_PATH)
    df = df.drop(["unparsed_text"],axis=1)
    df.to_csv("%s/data/nytimes_style_articles/parsed_only_articles_df.csv"%DATA_PATH)


def get_hand_curated_style_terms_articles_df():
    # load allowable fashion terms with labels for noun
    print ("TODO")


def add_tokenage_to_parsed(starter = "parsed_only_articles_df.csv"):
    print ("Starting from parsed only data")
    fin_df = pd.DataFrame(columns=['year','month','main_parts_text','type_of_material','web_url','headline','word_count','abstract','snippet','lead_paragraph','keywords','section_name','subsection_name','pub_date','nouns_in_main_parts','adjectives_in_main_parts','matched_keywords'])
    for chunk in pd.read_csv("%s/data/nytimes_style_articles/%s"%(DATA_PATH,starter), chunksize=50000):
        df = chunk
        df["main_parts_text"] = df.apply(get_main_parts, axis = 1)
        df["noun_phrases_in_main_parts"] = df.apply(get_noun_phrases, axis = 1) # headline, abstract, snipper, lead paragraph
        df["nouns_in_main_parts"] = df.apply(get_nouns, axis = 1)
        df["adjectives_in_main_parts"] = df.apply(get_adjectives, axis = 1)
        pd.concat([fin_df,df], ignore_index=True)
        del df
        gc.collect()
    """
    df = pd.read_csv("%s/data/nytimes_style_articles/%s"%(DATA_PATH,starter))
    df["main_parts_text"] = df.apply(get_main_parts, axis = 1)
    df["noun_phrases_in_main_parts"] = df.apply(get_noun_phrases, axis = 1) # headline, abstract, snipper, lead paragraph
    df["nouns_in_main_parts"] = df.apply(get_nouns, axis = 1)
    df["adjectives_in_main_parts"] = df.apply(get_adjectives, axis = 1)
    """
    fin_df.to_csv("%s/data/nytimes_style_articles/tokenaged_%s"%(DATA_PATH,starter))

#unparsed_to_parsed()
#parsed_to_parsed_without_unparsed_text()
add_tokenage_to_parsed()
