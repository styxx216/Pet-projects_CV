import sys
import os
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
import requests

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = 'cpu'

PATH_TO_MODEL = 'test_model.pth'
PATH_TO_FILES = sys.argv[1]

class dataset(Dataset):
    def __init__(self, path,transform=None):
        self.path = path
        self.transform = transform
        self.names = os.listdir(self.path)
    def __len__(self):
        res = len(self.names)
        return res
    def __getitem__(self,idx):
        path = os.path.join(self.path, self.names[idx])
        image = cv2.imread(path,0)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        image = torch.tensor(image).float().to(device)/255
        image = torch.unsqueeze(image, 0)
        image = torch.unsqueeze(image, 0)
        return image, path

img_set = dataset(PATH_TO_FILES)

n= len(img_set)



class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1,4,kernel_size=5,padding=2)#[24,24,1]->[24,24,4]
        self.bn_1 = torch.nn.BatchNorm2d(4)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.conv_2 = torch.nn.Conv2d(4,8,kernel_size=3,padding=1)#[12,12,4]->[12,12,8]
        self.bn_2 = torch.nn.BatchNorm2d(8)
        self.conv_3 = torch.nn.Conv2d(8,16,kernel_size=3,padding=1)#[6,6,8]->[6,6,16]
        self.bn_3 = torch.nn.BatchNorm2d(16)
        
        self.dropout = torch.nn.Dropout()
    def forward(self,x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.dropout(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.dropout(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = F.relu(x)
        x = self.pool(x)
        return x
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

        self.fc_1 = torch.nn.Linear(3*3*16, 256)
        self.fc_2 = torch.nn.Linear(256, 1)
        self.dropout = torch.nn.Dropout()
    def forward(self,x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc_1(x)
        x = torch.sigmoid(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        x = torch.sigmoid(x)

        return x
         

model = TestModel()
model.load_state_dict(torch.load(PATH_TO_MODEL))

pathes = []
labels = []

for i in range(4):
    pred = np.round(model(img_set[i][0]).detach().numpy())[0][0]
    pathes.append(img_set[i][1])
    labels.append(int(pred))

df = pd.DataFrame({'путь-до-файла':pathes, 'метка-класса':labels})

df.to_csv('./result.csv')
