import os 

ROOT = '/share/zhlu6105/dataset/360SP-data/train/panoramas_train'


with open('./SP360_trian.txt', 'w') as f:
    for img_name in os.listdir(ROOT):
        f.write(img_name+'\n')
    f.flush() 
