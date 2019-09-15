import pandas as pd

keywords_set = set([])
for l in open("data/keywords.txt"):
    keywords_set.add(l.strip())
print (keywords_set)
