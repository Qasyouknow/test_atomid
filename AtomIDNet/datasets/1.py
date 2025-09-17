import os
import csv

# 定义数据路径
data_dir = r'D:\PythonProjects\pythonProject\PolyU\FYP\AtomIDNet\datasets/syn'
csv_path = r'D:\PythonProjects\pythonProject\PolyU\FYP\AtomIDNet\datasets/tem_unet_train.csv'

# 获取所有 .bmp 文件
image_files = [f for f in os.listdir(data_dir) if f.endswith('_syn.bmp')]

# 写入 CSV 文件
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image'])  # 写入表头
    for image_file in image_files:
        writer.writerow([os.path.join('syn', image_file)])  # 相对路径