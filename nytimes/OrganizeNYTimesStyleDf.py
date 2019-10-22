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

def add_tokenage_to_parsed(starter = "parsed_only_articles_df.csv"):
    print ("Starting from parsed only data")
    fin_df = pd.DataFrame(columns=['year','month','main_parts_text','type_of_material','web_url','headline','word_count','abstract','snippet','lead_paragraph','keywords','section_name','subsection_name','pub_date','nouns_in_main_parts',"noun_phrases_in_main_parts",'adjectives_in_main_parts','matched_keywords'])
    count = 0
    for chunk in pd.read_csv("%s/data/nytimes_style_articles/%s"%(DATA_PATH,starter), chunksize=50000):
        print ("starting with chunk %d"%count)
        df = chunk
        df["main_parts_text"] = df.apply(get_main_parts, axis = 1)
        df["noun_phrases_in_main_parts"] = df.apply(get_noun_phrases, axis = 1) # headline, abstract, snipper, lead paragraph
        df["nouns_in_main_parts"] = df.apply(get_nouns, axis = 1)
        df["adjectives_in_main_parts"] = df.apply(get_adjectives, axis = 1)
        fin_df = pd.concat([fin_df,df], ignore_index=True)
        del df
        gc.collect()
        count +=1
    fin_df.to_csv("%s/data/nytimes_style_articles/tokenaged_%s"%(DATA_PATH,starter))

def to_keep_based_on_fashion_labels(row):
    global allowable_fashion_terms
    nouns_in_main_parts= [row.nouns_in_main_parts[1:-1].split(",")]
    nouns_in_main_parts =[el.strip()[1:-1].lower() for el in nouns_in_main_parts]
    nouns_phrases_in_main_parts= [el.strip()[1:-1].lower() for el in row.noun_phrases_in_main_parts[1:-1].split(",")]
    matched = []
    for term in allowable_fashion_terms:
        if term in nouns_in_main_parts or term in nouns_phrases_in_main_parts:
            matched.append(term.lower())
    return matched

def style_sec_true(row): #temporary measure before fixing section column
    if row.matched_keywords == None or len(row.matched_keywords) == 0:
        return False
    return "Style" in [el.strip()[1:-1] for el in row.matched_keywords[1:-1].split(",")]

def get_hand_curated_style_terms_articles_df():
    #  filter out bad rows in df that are not fashiony
    fin_df = pd.DataFrame(columns=['year','month','type_of_material','web_url','headline','word_count','abstract','snippet','lead_paragraph','keywords','section_name','subsection_name','pub_date','main_parts_text','nouns_in_main_parts','noun_phrases_in_main_parts','adjectives_in_main_parts','curated_matched_keyords'])
    count = 0
    for chunk in pd.read_csv("%s/data/nytimes_style_articles/tokenaged_parsed_only_articles_df.csv"%(DATA_PATH), chunksize=50000, low_memory=False):
        print ("starting with chunk %d"%count)
        df = chunk[['year','month','type_of_material','web_url','headline','word_count','abstract','snippet','lead_paragraph','keywords','section_name','subsection_name','pub_date','main_parts_text','nouns_in_main_parts','nouns_in_main_parts','noun_phrases_in_main_parts','adjectives_in_main_parts','matched_keywords']]
        # keep only rows that have appropriate style words OR that are style section
        df["style_sec_true"] = df.apply(style_sec_true, axis = 1) # TODO: rn the section name data is missing, need to debug, note that section_name can in fact be accessed thru json loads unparsed...
        df["curated_matched_keyords"] = df.apply(to_keep_based_on_fashion_labels, axis = 1)
        df =df[(df.astype(str)['curated_matched_keywords'] != '[]') | df["style_sec_true"] | (df['section_name'] == 'Style')]
        df.drop(['matched_keywords', 'style_sec_true'], axis=1)
        fin_df = pd.concat([fin_df,df], ignore_index=True)
        del df
        gc.collect()
        count +=1

    fin_df.to_csv("%s/data/nytimes_style_articles/curated_tokenaged_parsed_only_articles_df.csv"%(DATA_PATH))

def get_allowable_fashion_terms():
    FASHION_DATA_PATH = ""
    try:
        f=open("fashion_data_location.txt", "r")
        FASHION_DATA_PATH  = f.read().strip()
    except:
        print ("data is right here")
    fashion_df = pd.read_csv("%s/data/my_data/fashion_terms.csv"%(FASHION_DATA_PATH))
    fashion_list = fashion_df[['word','human_edited_label']].apply(list).values.tolist()
    allowable_fashion_terms = [r[0] for r in fashion_list if r[1] != 0]

#unparsed_to_parsed()
#parsed_to_parsed_without_unparsed_text()
#add_tokenage_to_parsed()
# load allowable fashion terms with labels for noun
allowable_fashion_terms = get_allowable_fashion_terms()
get_hand_curated_style_terms_articles_df()
