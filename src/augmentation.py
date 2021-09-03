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


train_input_path_list = image_path[0:]#We got the elements of the image_path_list list from 1905 to the last element
train_label_path_list = mask_path[0:]#We got the elements of the mask_path_list list from 1905 to the last element

for image in tqdm.tqdm(train_input_path_list):
    img=Image.open(image)
    color_aug = T.ColorJitter(brightness=0.6, contrast=0.4, hue=0.05)

    img_aug = color_aug(img)
    new_path=image[:-4]+"-1"+".png"
    new_path=new_path.replace('images', 'augmentation')
    img_aug=np.array(img_aug)
    cv2.imwrite(new_path,img_aug)
  
    
    # flipUD = np.flipud(img)
    # new2_path=image[:-4]+"-2"+".png"
    # new2_path=new2_path.replace('image', 'augmentation')
    # cv2.imwrite(new2_path,flipUD )

    
    # rotated = transform.rotate(img)
    
    # new3_path=image[:-4]+"-3"+".png"
    # new3_path=new3_path.replace('image', 'augmentation')
    # cv2.imwrite(new3_path,rotated)

    
    # transform = AffineTransform(translation=(25,25))
    # wrapShift = warp(img,transform,mode='wrap')
    # new4_path=image[:-4]+"-4"+".png"
    # new4_path=new4_path.replace('image', 'augmentation')
    # cv2.imwrite(new4_path,wrapShift)

    # sigma=0.155
    # #add random noise to the image
    # noisyRandom = random_noise(img,var=sigma**2)
    # plt.imshow(noisyRandom)
    # new5_path=image[:-4]+"-5"+".png"
    # new5_path=new5_path.replace('image', 'augmentation')
    # cv2.imwrite(new5_path,wrapShift)
    
    # noisyRandom=random_noise(img,mode='s&p',clip=True)
    # new5_path=image[:-4]+"-5"+".png"
    # new5_path=new5_path.replace('image', 'augmentation')
    # cv2.imwrite(new5_path,noisyRandom)
    

    
for mask in tqdm.tqdm(train_label_path_list):
    msk=cv2.imread(mask)
    new_mask=msk
    newm_path=mask[:-4]+"-1"+".png"
    newm_path=newm_path.replace('masks', 'augmentation_mask')
    cv2.imwrite(newm_path,new_mask)
    
    # flipUD = np.flipud(msk)
    # newm2_path=mask[:-4]+"-2"+".png"
    # newm2_path=newm2_path.replace('masks', 'augmentation_mask')
    # cv2.imwrite(newm2_path,flipUD )
    
    # rotated = transform.rotate(msk)
    # newm3_path=image[:-4]+"-3"+".png"
    # newm3_path=newm3_path.replace('masks', 'augmentation_mask')
    # cv2.imwrite(newm3_path,rotated)
    
    # transform = AffineTransform(translation=(25,25))
    # wrapShift = warp(msk,transform,mode='wrap')
    # newm4_path=mask[:-4]+"-4"+".png"
    # newm4_path=newm4_path.replace('masks', 'augmentation_mask')
    # cv2.imwrite(newm4_path,wrapShift)
    
    
    # sigma=0.155
    # #add random noise to the image
    # noisyRandom = random_noise(msk,var=sigma**2)
    # newm5_path=mask[:-4]+"-5"+".png"
    # newm5_path=newm5_path.replace('masks', 'augmentation_mask')
    # cv2.imwrite(newm5_path,wrapShift)
    
    # noisyRandom=img
    # newm5_path=image[:-4]+"-5"+".png"
    # newm5_path=newm5_path.replace('masks', 'augmentation_mask')
    # cv2.imwrite(newm5_path,noisyRandom)
    
    
    

    
