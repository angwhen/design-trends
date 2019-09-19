import pandas as pd
import json

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

def unparsed_to_parsed():
    df = pd.read_csv("data/nytimes_style_articles/unparsed_articles_df.csv")

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

    df.to_csv("data/nytimes_style_articles/parsed_articles_df.csv")

def parsed_to_parsed_without_unparsed_text():
    df = pd.read_csv("data/nytimes_style_articles/parsed_articles_df.csv")
    df = df.drop(["unparsed_text"],axis=1)
    df.to_csv("data/nytimes_style_articles/parsed_only_articles_df.csv")


parsed_to_parsed_without_unparsed_text()
