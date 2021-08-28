import torch
import glob
import tqdm
input_shape = (224, 224)
n_classes=2
import numpy as np
import cv2
import os
from preprocess import tensorize_image
from mask_on_image import write_mask_on_image


valid_size = 0.3#Validation dataset is used to evaluate a particular model, but this is for frequent evaluation.
test_size  = 1#rate of data to be tested
batch_size = 8#it means how many data the model will process at the same time.
epochs = 25#Epoch count is the number of times all training data is shown to the network during training.

input_shape = (224, 224)#What size will the image resize
n_classes = 2

model_path = 'model/model2.pt'
model = torch.load(model_path,map_location=torch.device('cpu'))
model.eval()

SRC_DIR = os.getcwd()#The method tells us the location of the current working directory (CWD).
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
IMAGE_DIR = os.path.join(DATA_DIR, 'test_data')


predict_file_path = '../data/predict_data/'
#for mask visualize from model output
# y=0
# for test_img in test:
#     tensorized_test_image = tensorize_image([test_img], input_shape, cuda)
#     output=model(tensorized_test_image)
#     print(output.shape)
#     torchvision.utils.save_image(output,('../data/model_image/'+str(y)+'.png'))
#     y+=1
test_data_list = os.listdir('../data/test_data/')
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()
#The names of the files in the IMAGE_DIR path are listed and sorted
mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()
indices = np.random.permutation(len(image_path_list))
test_ind  = int(len(indices) * test_size)#Multiply indices length by test_size and assign it to an int-shaped variable
valid_ind = int(test_ind + len(indices) * valid_size)
test_input_path_list = image_path_list[:test_ind] #Get 0 to 476 elements of the image_path_list list
test_label_path_list = mask_path_list[:test_ind]#Get 0 to 476 elements of the mask_path_list list
def predict(test_input_path_list):

    for i in tqdm.tqdm(range(len(test_input_path_list))):
        batch_test = test_input_path_list[i:i+1]
        test_input = tensorize_image(batch_test, input_shape)
        outs = model(test_input)
        out=torch.argmax(outs,axis=1)
        out_cpu = out.cpu()
        outputs_list=out_cpu.detach().numpy()
        mask=np.squeeze(outputs_list,axis=0)
            
        predict = predict_file_path + test_data_list[i]
            
        img=cv2.imread(batch_test[0])
        mg=cv2.resize(img,(224,224))
        cpy_img  = mg.copy()
        mg[mask==0 ,:] = (255, 0, 125)
        opac_image=(mg/2+cpy_img/2).astype(np.uint8)
        cv2.imwrite(predict,opac_image.astype(np.uint8))

predict(test_input_path_list)