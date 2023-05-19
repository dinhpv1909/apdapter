import json
import cv2
import os
from basicsr.utils import img2tensor
import pandas as pd


class Open_Pose_Dataset():
    def __init__(self, file_csv, data_path):
        super(Open_Pose_Dataset, self).__init__()

        self.files = []
        data = pd.read_csv(file_csv)
        for line in range(len(data)):
            img_path = f'{data_path}/{data["ID"][line]}/{data["Image"][line]}'
            open_pose_img_path = img_path
            txt_path = img_path + '.txt'
            self.files.append({'img_path': img_path, 'open_pose_img_path': open_pose_img_path, 'txt_path': txt_path})

    def __getitem__(self, idx):
        file = self.files[idx]

        im = cv2.imread(f"{file['img_path']}")
        h = im.shape[0]
        w = im.shape[1]
        
        if w % 8 != 0:
            w = w + (8- (w % 8))
        if h % 8 !=0:
            h = h+ (8- (h % 8))
            
        im = cv2.resize(im,(w,h), interpolation = cv2.INTER_AREA) 
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        open_pose = cv2.imread(f"{file['open_pose_img_path']}")  # [:,:,0]
        
        open_pose = cv2.resize(open_pose,(w,h), interpolation = cv2.INTER_AREA)
        open_pose = img2tensor(open_pose, bgr2rgb=True, float32=True) / 255.  # [0].unsqueeze(0)#/255.

        with open(f"{file['txt_path']}", 'r') as fs:
            sentence = fs.readline().strip()

        return {'im': im, 'open_pose': open_pose, 'sentence': sentence}
    

    def __len__(self):
        return len(self.files)