import time
import json
import pickle
import pandas as pd

all_years = []
for year in range(1904,2020): #1904 is cumulative when i ran it : (, correctly run can start from 1852
    print (year)
    curr = pickle.load(open("data/nytimes_style_articles/year_%d.p"%year,"rb"))
    all_years.extend(curr)

df=pd.DataFrame(all_years)
df.columns = ["year","month","unparsed_text","matched_keywords"]
df.to_csv("data/nytimes_style_articles/unparsed_articles_df.csv")
