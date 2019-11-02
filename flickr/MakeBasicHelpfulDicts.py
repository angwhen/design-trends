import pandas as pd
import pickle

DATA_PATH = "."
try:
    f=open("data_location.txt", "r")
    DATA_PATH  = f.read().strip()
except:
    print ("data is right here")

def make_fnum_to_flickr_url_dict(df):
    my_urls_list = df[["url","file_name"]].values.tolist()
    # Make filename num to url dict
    fnum_to_url_dict = {}
    for el in my_urls_list:
        url = el[0]
        p1 = url.split("_o")[0]
        url = p1+"_n.jpg"
        fname_num = el[2].split("/")[-1]
        fname_num = (int) (fname_num.split(".jpg")[0])
        fnum_to_url_dict[fname_num]=url
    pickle.dump(fnum_to_url_dict,open("%s/fnum_to_flickr_url_dict.p"%DATA_PATH,"wb"))

def make_fnum_to_year_dict(df):
    years_list = df[["file_name","year"]].values.tolist()
    fnum_to_year_dict = {}
    for el in years_list:
        fname_num = int(el[0].split(".")[0].split("/")[-1])
        year = (int)el[1]
        fnum_to_year_dict[fname_num] = year
    pickle.dump(fnum_to_year_dict,open("%s/fnum_to_year_dict.p"%DATA_PATH,"wb"))

df =  pd.read_csv("%s/data/url_title_and_file_data.csv"%DATA_PATH)
make_fnum_to_flickr_url_dict(df)
make_fnum_to_year_dict(df)
