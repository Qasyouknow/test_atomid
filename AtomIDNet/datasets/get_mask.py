import os
import csv
import cv2
import numpy as np

PATH = ".\\datasets\\tem"

def csv2array(path:str):
    array = []
    with open(path, 'r') as f:
        content = csv.reader(f)
        next(content)
        for line in content:
            array.append(line)
    
    return np.array(array)

def getFileList(path:str, endwith:str):
    iter = os.walk(path)
    path_list = []
    for p, d, filelist in iter:
        for name in filelist:
            if name.endswith(endwith):
                path_list.append(os.path.join(p, name))
                
    path_list.sort()
    return path_list

bmp_path_list = getFileList(PATH, '.bmp')
csv_path_list = getFileList(PATH, '.csv')

for img_path, csv_path in zip(bmp_path_list, csv_path_list):
    img = cv2.imread(img_path)
    points = csv2array(csv_path)[:,1:].astype('int')
    
    mask = np.zeros(img.shape)
    mask[points[:,1], points[:,0], -1] = 255
    
    mask_save_path = img_path.replace('.bmp', '_mask.png')
    cv2.imwrite(mask_save_path, mask.astype('uint8'))
        