import os
import random
import numpy as np
from PIL import Image

rootdir = "D:\\COCO-Text-words-trainval\\val_words"
mapping_file_path = "D:\\COCO-Text-words-trainval\\val_words_gt.txt"

file_names = []
for parent, dirnames, filenames in os.walk(rootdir):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
    file_names = filenames
    # for filename in filenames:                        #输出文件信息
    #     print("parent is" + parent)
    #     print("filename is:" + filename)
    #     print("the full name of the file is:" + os.path.join(parent, filename))


# read mapping as list
mapping_list = {}
map_file = open(mapping_file_path, encoding='utf-8')

for i in map_file:
    i = i.replace("\n", '')
    a = i.split(',')[0]
    b = i.replace(a+',', "")
    mapping_list[a] = b

def get_test_text_and_image(x ):
    # read a image and get a name than get mapping
    # x = random.randint(0, len(file_names) - 1)
    file_name = file_names[x]
    file_path = str(rootdir)
    if not file_path.endswith("\\"):
        file_path += "\\"
    file_path += file_name
    captcha_image = Image.open(file_path)
    captcha_image = captcha_image.resize((256, 64))
    captcha_image = np.array(captcha_image)

    text, image =  mapping_list[file_name.split(".")[0].split("_")[0]], captcha_image
    return text, image
