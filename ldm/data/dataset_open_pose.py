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
            img_path = f'{data_path}/{data["Image_path"][line]}'
            open_pose_img_path = f'{data_path}/{data["Open_Pose_path"][line]}'
            txt_path = f'{data_path}/{data["Caption_path"][line]}'
            self.files.append({'img_path': img_path, 'open_pose_img_path': open_pose_img_path, 'txt_path': txt_path})

    def __getitem__(self, idx):
        file = self.files[idx]
        h,w = 896,512
        

        im = cv2.imread(f"{file['img_path']}")           
            
        im = cv2.resize(im,(w,h), interpolation = cv2.INTER_AREA) # 3:4 config to dataset tik tok
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        open_pose = cv2.imread(f"{file['open_pose_img_path']}")  # [:,:,0]
        
        open_pose = cv2.resize(open_pose,(w,h), interpolation = cv2.INTER_AREA) # 3:4 config to dataset tik tok
        open_pose = img2tensor(open_pose, bgr2rgb=True, float32=True) / 255.  # [0].unsqueeze(0)#/255.

        with open(f"{file['txt_path']}", 'r') as fs:
            sentence = fs.readline().strip()

        return {'im': im, 'open_pose': open_pose, 'sentence': sentence}
    

    def __len__(self):
        return len(self.files)