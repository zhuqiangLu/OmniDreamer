import os 

ROOT = '/share/zhlu6105/dataset/video_frames/train'


with open('./train.txt', 'w') as f:
    for video_dir in os.listdir(ROOT):
        for img in os.listdir(os.path.join(ROOT, video_dir,)):
            f.write(video_dir + "/" + img+'\n')
    f.flush() 
