import pandas as pd


df =  pd.read_csv("data/url_title_and_file_data.csv")
my_list = df[["file_name","year"]].values.tolist()
my_list.sort(key=lambda x: x[1])




my_str = ""
for im in my_list:
    fname_specific = im[0].split("/")[-1]
    fname_num = fname_specific.split(".")[0]
    my_str += "import im%s from '../flickr_images/default_images/%s';\n"%(fname_num,fname_specific)

my_str += "\n"
my_str += "  images: [\n"
for im in my_list[:-2]:
    fname_specific = im[0].split("/")[-1]
    fname_num = fname_specific.split(".")[0]
    my_str += "[im%s,'%s'],\n"%(fname_num,im[1])
fname_specific = im[0].split("/")[-1]
fname_num = fname_specific.split(".")[0]
my_str += "[im%s,'%s']\n"%(fname_num , my_list[:-1][1])
my_str += "],"

text_file = open("data/react_default_images_timeline_code_lists_and_imports.txt", "w")
text_file.write(my_str)
text_file.close()
