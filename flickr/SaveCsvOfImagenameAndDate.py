import pandas as pd

keywords_set = set([])
for l in open("data/keywords.txt"):
    keywords_set.add(l.strip())
print (keywords_set)

df = None
for keyword in keywords_set:
    if df is None:
        df =  pd.read_csv("data/%s_urls.csv"%keyword)
    else:
        df = df.append(pd.read_csv("data/%s_urls.csv"%keyword))

df = df.drop_duplicates(subset=['url','title'], keep="first")

url_to_image_file_df = pd.read_csv("data/images/url_to_image_file.csv")

res_df = pd.merge(df, url_to_image_file_df, on='url', how='inner')

res_df = res_df[["url","title","year","file_name"]]

res_df.to_csv("data/url_title_and_file_data.csv")
