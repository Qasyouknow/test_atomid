import csv
import os
import numpy as np

def getFilelist(path:str, endwith:str):
    iter = os.walk(path)
    path_list = []
    for p, d, filelist in iter:
        for name in filelist:
            if name.endswith(endwith):
                path_list.append(os.path.join(p, name))
    path_list.sort()
    return path_list

PATH = '.\\tem'
image_list = getFilelist(PATH, 'bmp')
label_list = getFilelist(PATH, 'csv')
print(image_list)
indices = np.arange(len(image_list))
np.random.shuffle(indices)

image_train = np.array(image_list)[indices][:int(0.6*len(image_list))]
label_train = np.array(label_list)[indices][:int(0.6*len(image_list))]
image_val = np.array(image_list)[indices][int(0.6*len(image_list)):int(0.8*len(image_list))]
label_val = np.array(label_list)[indices][int(0.6*len(image_list)):int(0.8*len(image_list))]
image_test = np.array(image_list)[indices][int(0.8*len(image_list)):]
label_test = np.array(label_list)[indices][int(0.8*len(image_list)):]

with open("tem_unet_train.csv", "w+", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["index", "image", "csv"])
    for idx in range(len(image_train)):
        writer.writerow([idx+1, image_train[idx].split('uc\\')[-1], label_train[idx].split('uc\\')[-1]])

with open("tem_unet_val.csv", "w+", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["index", "image", "csv"])
    for idx in range(len(image_val)):
        writer.writerow([idx+1, image_val[idx].split('uc\\')[-1], label_val[idx].split('uc\\')[-1]])

with open("tem_unet_test.csv", "w+", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["index", "image", "csv"])
    for idx in range(len(image_val)):
        writer.writerow([idx+1, image_test[idx].split('uc\\')[-1], label_test[idx].split('uc\\')[-1]])
    