import pandas as pd
import json

df = pd.read_csv("data/nytimes_style_articles/parsed_only_articles_df.csv")
df2  = df[df.matched_keywords.apply(lambda x: 'dress' in x)][["year","month","snippet","lead_paragraph","abstract"]]
df2 = df2.dropna(thresh=3)
print (df2.head(50))
