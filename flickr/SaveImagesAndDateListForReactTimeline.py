import pandas as pd


def get_images_code_for_react():
    df =  pd.read_csv("data/url_title_and_file_data.csv")
    my_list = df[["url","year"]].values.tolist()
    my_list.sort(key=lambda x: x[1])

    #TODO: want to also filter out images that are not clothes/do not have people


    my_str = ""
    my_str += "  images: [\n"
    for im in my_list[:-2]:
        fname_specific = im[0].split("/")[-1]
        fname_num = fname_specific.split(".")[0]
        my_str += "[im%s,'%s'],\n"%(fname_num,im[1])
    fname_specific = my_list[-1][0].split("/")[-1]
    fname_num = fname_specific.split(".")[0]
    my_str += "[im%s,'%s']\n"%(fname_num , my_list[-1][1])
    my_str += "],"

    text_file = open("data/react_default_images_timeline_code_lists_and_imports.txt", "w")
    text_file.write(my_str)
    text_file.close()

get_images_code_for_react()
