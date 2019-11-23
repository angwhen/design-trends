import pandas as pd
import pickle

try:
    DATA_PATH  = open("data_location.txt", "r").read().strip()
except:
    DATA_PATH = "."
try:
    VFG_DATA_PATH  = open("../vintage_fashion_guild/data_location.txt", "r").read().strip()
except:
    VFG_DATA_PATH = "../vintage_fashion_guild"

def make_fnum_to_flickr_url_dict(df):
    print ("making url dict")
    my_urls_list = df[["url","file_name"]].values.tolist()
    # Make filename num to url dict
    fnum_to_url_dict = {}
    for el in my_urls_list:
        url = el[0]
        p1 = url.split("_o")[0]
        url = p1+"_n.jpg"
        fname_num = el[1].split("/")[-1]
        fname_num = (int) (fname_num.split(".jpg")[0])
        fnum_to_url_dict[fname_num]=url
    pickle.dump(fnum_to_url_dict,open("%s/data/basics/fnum_to_flickr_url_dict.p"%DATA_PATH,"wb"))
    try:
        vfg_fnum_to_url_dict = pickle.load(open("%s/data/vfg_fnum_to_url_dict.p"%VFG_DATA_PATH,"rb"))
        for fnum in vfg_fnum_to_url_dict.keys():
            fnum_to_url_dict[fnum]=url
        pickle.dump(fnum_to_url_dict,open("%s/data/basics/fnum_to_url_dict.p"%DATA_PATH,"wb"))
    except:
        print ("do not have vintage fashion guild data yet")

def make_fnum_to_year_dict(df):
    print ("making fnum to year dict")
    years_list = df[["file_name","year"]].values.tolist()
    fnum_to_year_dict = {}
    for el in years_list:
        fname_num = int(el[0].split(".")[0].split("/")[-1])
        year = int(el[1])
        fnum_to_year_dict[fname_num] = year

    try:
        vfg_fnum_to_year_dict = pickle.load(open("%s/data/vfg_fnum_to_year_dict.p"%VFG_DATA_PATH,"rb"))
        for fnum in vfg_fnum_to_year_dict.keys():
            fnum_to_year_dict[fnum] = vfg_fnum_to_year_dict[fnum]
    except:
        print ("do not have vintage fashion guild data yet")
    pickle.dump(fnum_to_year_dict,open("%s/data/basics/fnum_to_year_dict.p"%DATA_PATH,"wb"))

def make_year_to_fnums_dict(df):
    print ("making year to fnums dict")
    years_list = df[["file_name","year"]].values.tolist()
    year_to_fnums_dict = {}
    for el in years_list:
        fnum = int(el[0].split(".")[0].split("/")[-1])
        year = int(el[1])
        if year not in year_to_fnums_dict:
            year_to_fnums_dict[year] = []
        year_to_fnums_dict[year].append(fnum)
    try:
        # TODO: accidently deleted this file, may want to recover
        vfg_year_to_fnums_dict = pickle.load(open("%s/data/vfg_year_to_fnums_dict.p"%VFG_DATA_PATH,"rb"))
        for year in vfg_year_to_fnums_dict .keys():
            if year not in year_to_fnums_dict:
                year_to_fnums_dict[year] = vfg_year_to_fnums_dict [year]
            year_to_fnums_dict[year].extend(vfg_year_to_fnums_dict[year])
    except:
        print ("do not have vintage fashion guild data yet")
    pickle.dump(year_to_fnums_dict,open("%s/data/basics/year_to_fnums_dict.p"%DATA_PATH,"wb"))

def make_fnums_list(df):
    print ("making fnums list")
    my_list = df[["file_name"]].values.tolist()
    fnums_list = []
    for el in my_list:
        fname_num = int(el[0].split(".")[0].split("/")[-1])
        fnums_list.append(fname_num)
    pickle.dump(fnums_list,open("%s/data/basics/flickr_fnums_list.p"%DATA_PATH,"wb"))
    # add vintage fashion data
    try:
        fnums_list.extend( pickle.load(open("%s/data/vfg_fnums_list.p"%VFG_DATA_PATH,"rb")) )
    except:
        print ("do not have vintage fashion guild data yet")
    pickle.dump(list(set(fnums_list)),open("%s/data/basics/fnums_list.p"%DATA_PATH,"wb"))

df =  pd.read_csv("%s/data/url_title_and_file_data.csv"%DATA_PATH)
make_fnum_to_flickr_url_dict(df)
make_fnum_to_year_dict(df)
make_fnums_list(df)
make_year_to_fnums_dict(df)
