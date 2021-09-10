import numpy as np
import cv2
import json
import os
import torch
import tqdm 
from matplotlib import pyplot as plt
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage import transform
import random 
from torchvision import transforms as T
from PIL import Image
from skimage import io
from PIL import ImageOps

IMAGE_DIR="../data/images"
#The path to the image folder is assigned to the variable
MASK_DIR="../data/masks"


#The path to the masks folder is assigned to the variable
image_path=[] #empty list created
for name in os.listdir(IMAGE_DIR):
    image_path.append(os.path.join(IMAGE_DIR,name))
mask_path=[] #empty list created


for name in os.listdir(MASK_DIR):
    mask_path.append(os.path.join(MASK_DIR,name))
image_path.sort()
mask_path.sort()
valid_size = 0.30
indices = np.random.permutation(len(image_path))
valid_ind = int(len(indices) * valid_size)
train_input_path_list = image_path[valid_ind:]#We got the elements of the image_path_list list from 1905 to the last element
train_label_path_list = mask_path[valid_ind:]#We got the elements of the mask_path_list list from 1905 to the last element


for image in tqdm.tqdm(train_input_path_list):
    '''
    img=Image.open(image) 
    img_aug = T.functional.adjust_brightness(img,brightness_factor=0.5)
    #imgX = np.flip(img_aug, axis=1)
    color_aug = T.ColorJitter(brightness=0.4, contrast=0.4, hue=0.06)
    new3_path=image[:-4]+"-1"+".jpg"
    new3_path=new3_path.replace('images', 'augmentation2')
    imgconvert.save(new3_path)
    '''
    img=Image.open(image)
    color_aug = T.ColorJitter(brightness=0.4, contrast=0.4, hue=0.06)

    img_aug = color_aug(img)
    new_path=image[:-4]+"-1"+".png"
    new_path=new_path.replace('images', 'augmentation2')
    img_aug=np.array(img_aug)
    cv2.imwrite(new_path,img_aug)
    
for mask in tqdm.tqdm(train_label_path_list):
    msk=cv2.imread(mask)
    #maskX = np.flip(msk, axis=1) //ters çevirmek istersek
    newm2_path=mask[:-4]+"-1"+".png"
    newm2_path=newm2_path.replace('masks', 'augmentation_mask2')
    cv2.imwrite(newm2_path,maskX )
    

    

    
