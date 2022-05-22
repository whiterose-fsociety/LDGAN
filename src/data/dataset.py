import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as TF
from glob import glob,iglob
import config.config as config 
import torchvision
from torch.utils.data import Dataset,DataLoader
from PIL import Image

class ImageFolder(Dataset):
    def __init__(self,datatype="train"):
        self.datatype = datatype
        if self.datatype == "train":
            self.lr_dataset = sorted(glob(os.path.join(config.lr_train_dataset_folder_name,"*.png")))
            self.hr_dataset = sorted(glob(os.path.join(config.hr_train_dataset_folder_name,"*.png")))
        elif self.datatype == "valid":
            self.lr_dataset = sorted(glob(os.path.join(config.lr_valid_dataset_folder_name,"*.png")))
            self.hr_dataset = sorted(glob(os.path.join(config.hr_valid_dataset_folder_name,"*.png")))
        elif self.datatype == "test":
            self.lr_dataset = sorted(glob(os.path.join(config.lr_test_dataset_folder_name,"*.png")))
            self.hr_dataset = sorted(glob(os.path.join(config.hr_test_dataset_folder_name,"*.png")))
    
    def __len__(self):
        return len(self.hr_dataset)
    
    def __getitem__(self,index):
        lr_image = np.asarray(Image.open(self.lr_dataset[index]))
        hr_image = np.asarray(Image.open(self.hr_dataset[index]))
        
        hr_image = config.test_transform(image=hr_image)['image']
        lr_image = config.test_transform(image=lr_image)['image']
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(hr_image,output_size=(config.high_res,config.high_res)) # Get random crop of tensor hr image
        hr_pil = torchvision.transforms.ToPILImage()(hr_image) # Pil Version of hr_image
        lr_pil = torchvision.transforms.ToPILImage()(lr_image) # Pil Version of lr_image
        hcrop = TF.crop(hr_pil,i,j,h,w) # Get 128x128 crop of hr image
        lcrop = TF.crop(lr_pil,i//4,j//4,h//4,w//4) # Get 128x128 crop of lr image
        hcrop_tensor = config.test_transform(image=np.asarray(hcrop))['image'] # Tensor version of 128 crophcrop_tensor
        lcrop_tensor = config.test_transform(image=np.asarray(lcrop))['image'] # Tensor version of 128 crop
        if self.datatype != "test":
            return lcrop_tensor.float(),hcrop_tensor.float()
        else:
            return lr_image,hr_image
        
