import json
import cv2
import os
from basicsr.utils import img2tensor
import pandas as pd

class DepthDataset():
    def __init__(self, meta_file):
        super(DepthDataset, self).__init__()

        self.files = []
        data = pd.read_csv(meta_file)
        for line in range(len(data)):
            img_path = f'{data["ID"][line]}/{data["Image"][line]}'
            print(img_path)
            depth_img_path = img_path
            txt_path = img_path.rsplit('.', 1)[0] + '.txt'
            self.files.append({'img_path': img_path, 'depth_img_path': depth_img_path, 'txt_path': txt_path})

    def __getitem__(self, idx):
        file = self.files[idx]

        im = cv2.imread(f"/kaggle/input/datatiktok/control_pose_per_vid/{file['img_path']}")
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        depth = cv2.imread(f"/kaggle/input/datatiktok/control_pose_per_vid/{file['depth_img_path']}")  # [:,:,0]
        depth = img2tensor(depth, bgr2rgb=True, float32=True) / 255.  # [0].unsqueeze(0)#/255.

        with open(f"/kaggle/input/datatiktok/control_pose_per_vid/{file['txt_path']}", 'r') as fs:
            sentence = fs.readline().strip()

        return {'im': im, 'depth': depth, 'sentence': sentence}

    def __len__(self):
        return len(self.files)