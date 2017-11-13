# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 18:33:18 2017

@author: Sreenivas
"""

from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torch
import os
import os.path
import pandas as pd
from skimage import io,transform
class Imagedataset(Dataset):
    #Init function taking csv files for labels and root directory for images
    def __init__(self,root_dir,csv_file,transform=None):
        self.labelfile=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform
    #Length of dataset
    def __len__(self):
        return len(self.labelfile)
    #forming tuple of image and its label
    def __getitem__(self,idx):
        img_name=os.path.join(self.root_dir,self.labelfile.ix[idx,0])
        image=io.imread(img_name)
        label=self.labelfile.ix[idx,1]
        if self.transform:
            image=self.transform(image)
        return image,label
    
        
        
    
    
