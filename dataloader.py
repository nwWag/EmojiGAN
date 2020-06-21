import numpy as np
import cv2
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class EmojiDataset(Dataset):

    def __init__(self):
        datapath = 'data/'
        dirs = [dirname for _, dirname, _ in os.walk(datapath)][0]
        files = []
        with open(datapath+'emoji_pretty.json') as json_file:
            data = json.load(json_file)
            for entry in data:
                if entry['category'] == 'Smileys & Emotion' or entry['category'] == 'People & Body':
                    files.append(entry['image'])
        images = []
        for file in files:
            for dir in dirs:
                # print('read', datapath + dir + '/' + file)
                im = np.array(cv2.imread(datapath + dir + '/' + file, cv2.IMREAD_UNCHANGED))
                try:
                    im_rgba = im / 255.0
                    im_rgba = np.moveaxis(im_rgba, -1, 0)
                    if im_rgba.shape[0] == 3:
                        continue
                    images.append(im_rgba)
                except:
                    continue

        self.data = np.array(images)
        print(self.data.shape)
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.zeros(1)

if __name__ == '__main__':
    emj = EmojiDataset()
    img = emj.__getitem__(3)
    import matplotlib.pyplot as plt
    plt.imshow(img)