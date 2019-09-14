import time
import json
import pickle
import pandas

all_years = []
for year in range(2019,2020): #start 1852
    curr = pickle.load(open("data/nytimes_style_articles/year_%d.p"%year,"rb"))
    all_years.extend(curr)

df=pd.DataFrame(all_years)
df.columns = ["year","month","text","matched_keywords"]
