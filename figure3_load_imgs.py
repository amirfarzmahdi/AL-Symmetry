# load our stimulus set, natural images, and white noise images

# useful packages:
import scipy
import scipy.io as sio
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

# set random seed
np.random.seed(42)

# number of images
num_imgs = 2025
img_size = (224,224)

# our stimulus set
imgs_mat = scipy.io.loadmat('imgs.mat')
imgs = list(imgs_mat.values())
imgs = imgs[3].T
imgs = np.reshape(imgs,(num_imgs,1))
our_imgs = []
for i_image in range(num_imgs):
    img = imgs[i_image][0]
    img = np.stack((img,img,img),axis=2)
    our_imgs.append(img)

# natural images
test_image_dir =  'Datasets/ILSVRC2013/test'
names = os.listdir(test_image_dir)
names_idx = np.random.permutation(len(names))
selected_names = names[0:num_imgs]
natural_imgs = []
for _,filename in enumerate(selected_names):
    img = Image.open(os.path.join(test_image_dir,filename)).convert('RGB')
    img = np.asarray(img.resize(img_size))
    natural_imgs.append(img)

# white noise images
random_imgs = np.random.randint(low=0,high=256,size=(num_imgs,img_size[0],img_size[1],3)).astype('uint8')
random_imgs = list(random_imgs)

# save data dictionary
data_dict = {'our_imgs':our_imgs,
     'natural_imgs':natural_imgs,
     'random_imgs':random_imgs}

# save imges
with open('figure3_imgs.csv','wb') as fp:
    pickle.dump(data_dict,fp)