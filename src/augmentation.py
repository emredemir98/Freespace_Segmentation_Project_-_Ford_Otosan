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
MASK_DIR="../data/masks"

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
train_input_path_list = image_path[valid_ind:]
train_label_path_list = mask_path[valid_ind:]


for image in tqdm.tqdm(train_input_path_list):
    
    img=Image.open(image) 
    img_aug = T.functional.adjust_brightness(img,brightness_factor=0.5)
    imgX = np.flip(img_aug, axis=1)
    new3_path=image[:-4]+"-1"+".jpg"
    new3_path=new3_path.replace('images', 'augmentation')
    imgX.save(new3_path)
    
    
    
for mask in tqdm.tqdm(train_label_path_list):
    msk=cv2.imread(mask)
    maskX = np.flip(msk, axis=1) 
    newm2_path=mask[:-4]+"-1"+".png"
    newm2_path=newm2_path.replace('masks', 'augmentation_mask')
    cv2.imwrite(newm2_path,maskX )
    

    

    
