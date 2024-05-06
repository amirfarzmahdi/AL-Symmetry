# function for measuring horizontal, vertical, rotation invariance and equivariance
# useful packages:
import torch
import torchvision.transforms as transforms
from torchvision import models
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")
import pickle
from PIL import Image
import seaborn as sns

# check available GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# start settings:
nexemplar = 25
nview = 9
nclass = 9
num_imgs = nexemplar * nview * nclass
nconv = 13 # number of conv layers
nfc = 7 # number of fully-connected layers
layers = [1,4,7,9,11,15,18]
nconv_relu = 5
nfc_relu = 2
n_relu = nconv_relu + nfc_relu # number of relu layers

layer_out_original_images = [[] for i in range(n_relu)]
layer_out_hflipped_images = [[] for i in range(n_relu)]
layer_out_vflipped_images = [[] for i in range(n_relu)]
layer_out_90rotated_images = [[] for i in range(n_relu)]
layer_out_hfmaps_hflipped_images = [[] for i in range(n_relu)]
layer_out_vfmaps_vflipped_images = [[] for i in range(n_relu)]
layer_out_rfmaps_90rotated_images = [[] for i in range(n_relu)]

r_himgs = np.zeros((num_imgs,n_relu))
r_vimgs = np.zeros((num_imgs,n_relu))
r_r90imgs = np.zeros((num_imgs,n_relu))
r_himgs_hfmap = np.zeros((num_imgs,nconv_relu))
r_vimgs_vfmap = np.zeros((num_imgs,nconv_relu))
r_r90imgs_rfmap = np.zeros((num_imgs,nconv_relu))

# load images
with open('figure3_imgs.csv','rb') as fp:
    data_dict = pickle.load(fp)

cond = 'white_noise_trained' # condition of the network: 
        # '3D_objects_trained',
        # 'ImageNet_trained',
        # 'white_noise_trained',
        # '3D_objects_untrained',
        # 'ImageNet_untrained',
        # 'white_noise_untrained',
figname = 'figure3E'
train_condition = True # pretrained flag
input_images = data_dict['random_imgs'] # our_imgs, natural_imgs, random_imgs
# end settings


# load the network
alexnet = models.alexnet(pretrained = train_condition)
alexnet.eval()
alexnet.to(device)

# get activation function
activation = {}
def get_activation(name):
    def hook(net, input, output):
        activation[name] = output.detach()
    return hook

# image transforms:
# original
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# horizontally flipped
transform_h_flipped = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(1),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# vertically flipped
transform_v_flipped = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomVerticalFlip(1),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# 90-degree rotation
transform_90d_rotated = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomRotation((90,90)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform_image(image,transform):
    image_ = transform(image)
    image_ = torch.unsqueeze(image_,0)
    image_ = image_.to(device)
    return image_

def layer_output_flatten_squeeze(input_array):
    output_array = torch.flatten(input_array,1)
    output_array = output_array.squeeze()
    output_array = output_array.cpu().numpy()
    return output_array

def conv_layer_stack_flatten_squeeze(input_array):
    output_array = np.stack(input_array)
    output_array = output_array.flatten()
    output_array = output_array.squeeze()
    return output_array

# layer output in response to transformed images
def inv_equ_func(images,network):
    with torch.no_grad():
        activation_shape = []
        for i_image in range(0,len(images)):
            print(['image: ', i_image])

            #original image
            img = images[i_image]
            img_ = Image.fromarray(img)

            orig_img = transform_image(img_,transform)

            #horizontal filpped image
            hflipped_img = transform_image(img_,transform_h_flipped)

            #vertical filpped image
            vflipped_img = transform_image(img_,transform_v_flipped)

            #90 degree rotated image
            rotated90_img = transform_image(img_,transform_90d_rotated)
            
            num1 = -1
            for i in (layers):
                num1 = num1 + 1
                if i < nconv:
                    name_layer = 'features['+str(i)+']'
                    num_layer = i
                    alexnet.features[num_layer].register_forward_hook(get_activation(name_layer))
                else:
                    name_layer = 'classifier['+str(i-13)+']'
                    num_layer = i - nconv
                    alexnet.classifier[num_layer].register_forward_hook(get_activation(name_layer))

                #original image
                _ = alexnet(orig_img)
                layer_out_orig_img_raw = activation[name_layer]

                # activation map shape
                if i_image ==0:
                    activation_shape.append(layer_out_orig_img_raw.squeeze(0).shape)

                layer_out_orig_img = layer_output_flatten_squeeze(layer_out_orig_img_raw)
                layer_out_original_images[num1].append(layer_out_orig_img)
                
                #horizontal filpped image
                _ = alexnet(hflipped_img)
                layer_out_hflipped_img_raw = activation[name_layer]
                layer_out_hflipped_img = layer_output_flatten_squeeze(layer_out_hflipped_img_raw)
                layer_out_hflipped_images[num1].append(layer_out_hflipped_img)
                
                #vertical filpped image
                _ = alexnet(vflipped_img)
                layer_out_vflipped_img_raw = activation[name_layer]
                layer_out_vflipped_img = layer_output_flatten_squeeze(layer_out_vflipped_img_raw)
                layer_out_vflipped_images[num1].append(layer_out_vflipped_img)

                #90 degree rotated image
                _ = alexnet(rotated90_img)
                layer_out_90rotated_img_raw = activation[name_layer]
                layer_out_90rotated_img = layer_output_flatten_squeeze(layer_out_90rotated_img_raw)
                layer_out_90rotated_images[num1].append(layer_out_90rotated_img)
                            
                #flipping and rotating the feature maps

                #change feature maps
                #h flipped
                fmaps_hflipped_img = layer_out_hflipped_img_raw[0].cpu().numpy()
                hfmaps_hflipped_img = []

                # v flipped
                fmaps_vflipped_img = layer_out_vflipped_img_raw[0].cpu().numpy()
                vfmaps_vflipped_img = []

                # 90 rotated
                fmaps_90rotated_img = layer_out_90rotated_img_raw[0].cpu().numpy()
                rfmaps_90rotated_img = []

                if i < nconv:
                    for ch in range(0,len(fmaps_hflipped_img)):
                        # h flipped
                        tmp = fmaps_hflipped_img[ch,:,:]
                        hflipped_ch = np.fliplr(tmp)
                        hfmaps_hflipped_img.append(hflipped_ch)

                        # v flipped
                        tmp = fmaps_vflipped_img[ch,:,:]
                        vflipped_ch = np.flipud(tmp)
                        vfmaps_vflipped_img.append(vflipped_ch)

                        # 90 rotated
                        tmp = fmaps_90rotated_img[ch,:,:]
                        rflipped_ch = np.rot90(tmp)
                        rfmaps_90rotated_img.append(rflipped_ch)

                    # hflipped_fmap
                    hfmaps_hflipped_img = conv_layer_stack_flatten_squeeze(hfmaps_hflipped_img)
                    layer_out_hfmaps_hflipped_images[num1].append(hfmaps_hflipped_img)

                    # vflipped_fmap
                    vfmaps_vflipped_img = conv_layer_stack_flatten_squeeze(vfmaps_vflipped_img)
                    layer_out_vfmaps_vflipped_images[num1].append(vfmaps_vflipped_img)

                    # rotated_fmap
                    rfmaps_90rotated_img = conv_layer_stack_flatten_squeeze(rfmaps_90rotated_img)
                    layer_out_rfmaps_90rotated_images[num1].append(rfmaps_90rotated_img)

    return layer_out_original_images, layer_out_hflipped_images, layer_out_vflipped_images,\
            layer_out_90rotated_images, layer_out_hfmaps_hflipped_images, layer_out_vfmaps_vflipped_images,\
            layer_out_rfmaps_90rotated_images, activation_shape
        
# measure network outputs for different image/fmap transformations
layer_out_orig, layer_out_hflipped, layer_out_vflipped, layer_out_90rotated, hfmaps_hflipped,\
             vfmaps_vflipped, rfmaps_90rotated,activation_shape = inv_equ_func(input_images,alexnet)

# measure zscore and pearson corrleation
for i in range(0,len(layers)):
    layer_out_orig_z = stats.zscore(layer_out_orig[i])
    layer_out_hflipped_z = stats.zscore(layer_out_hflipped[i])
    layer_out_vflipped_z = stats.zscore(layer_out_vflipped[i])
    layer_out_90rotated_z = stats.zscore(layer_out_90rotated[i])
    
    hfmaps_hflipped_z = stats.zscore(hfmaps_hflipped[i])
    vfmaps_vflipped_z = stats.zscore(vfmaps_vflipped[i])
    rfmaps_90rotated_z = stats.zscore(rfmaps_90rotated[i])

    layer_shape = activation_shape[i] # activation shape
    print(layer_shape)

    # masks
    if i < nconv_relu:
        # remove centeral row
        tensor_wo_center_row = torch.ones(layer_shape,dtype=bool)
        tensor_wo_center_row[:,int(tensor_wo_center_row.shape[2]/2),:] = False
        tensor_wo_center_row = tensor_wo_center_row.flatten()
        # remove centeral column
        tensor_wo_center_col = torch.ones(layer_shape,dtype=bool)
        tensor_wo_center_col[:,:,int(tensor_wo_center_col.shape[1]/2)] = False
        tensor_wo_center_col = tensor_wo_center_col.flatten()
        # remove centeral pixel
        tensor_wo_center_pix = torch.ones(layer_shape,dtype=bool)
        tensor_wo_center_pix[:,int(tensor_wo_center_pix.shape[1]/2),int(tensor_wo_center_pix.shape[2]/2)] = False
        tensor_wo_center_pix = tensor_wo_center_pix.flatten()
    else:
        tensor_wo_center_row = tensor_wo_center_col = tensor_wo_center_pix = torch.ones(layer_shape[0],dtype=bool)

    for j in range(0,layer_out_orig_z.shape[0]):
        x = layer_out_orig_z[j,:]

        # horizontal flipped images
        x_wo_cent_col = torch.masked_select(torch.from_numpy(x),tensor_wo_center_col)
        x_wo_cent_col = x_wo_cent_col.numpy()
        y1 = layer_out_hflipped_z[j,:]
        y1_wo_cent_col = torch.masked_select(torch.from_numpy(y1),tensor_wo_center_col)
        y1_wo_cent_col = y1_wo_cent_col.numpy()

        nas = np.logical_or(np.isnan(x_wo_cent_col),np.isnan(y1_wo_cent_col))

        r_himg, _ = pearsonr(x_wo_cent_col[~nas], y1_wo_cent_col[~nas])
        r_himgs[j,i] = r_himg
        
        # vertical flipped images
        x_wo_cent_row = torch.masked_select(torch.from_numpy(x),tensor_wo_center_row)
        x_wo_cent_row = x_wo_cent_row.numpy()
        y2 = layer_out_vflipped_z[j,:]
        y2_wo_cent_row = torch.masked_select(torch.from_numpy(y2),tensor_wo_center_row)
        y2_wo_cent_row = y2_wo_cent_row.numpy()

        nas = np.logical_or(np.isnan(x_wo_cent_row),np.isnan(y2_wo_cent_row))

        r_vimg, _ = pearsonr(x_wo_cent_row[~nas], y2_wo_cent_row[~nas])
        r_vimgs[j,i] = r_vimg
        
        # 90 degree rotated images
        x_wo_cent_pix = torch.masked_select(torch.from_numpy(x),tensor_wo_center_pix)
        x_wo_cent_pix = x_wo_cent_pix.numpy()
        y3 = layer_out_90rotated_z[j,:]
        y3_wo_cent_pix = torch.masked_select(torch.from_numpy(y3),tensor_wo_center_pix)
        y3_wo_cent_pix = y3_wo_cent_pix.numpy()

        nas = np.logical_or(np.isnan(x_wo_cent_pix),np.isnan(y3_wo_cent_pix))

        r_r90img, _ = pearsonr(x_wo_cent_pix[~nas], y3_wo_cent_pix[~nas])
        r_r90imgs[j,i] = r_r90img

        if i < nconv_relu:
            # horizontally flipped images flipped res
            y4 = hfmaps_hflipped_z[j,:]
            y4_wo_cent_col = torch.masked_select(torch.from_numpy(y4),tensor_wo_center_col)
            y4_wo_cent_col = y4_wo_cent_col.numpy()

            nas = np.logical_or(np.isnan(x_wo_cent_col),np.isnan(y4_wo_cent_col))
            r_himg_hfmap, _ = pearsonr(x_wo_cent_col[~nas], y4_wo_cent_col[~nas])
            r_himgs_hfmap[j,i] = r_himg_hfmap

            # vertically flipped images flipped res
            y5 = vfmaps_vflipped_z[j,:]
            y5_wo_cent_row = torch.masked_select(torch.from_numpy(y5),tensor_wo_center_row)
            y5_wo_cent_row = y5_wo_cent_row.numpy()

            nas = np.logical_or(np.isnan(x_wo_cent_row),np.isnan(y5_wo_cent_row))
            r_vimg_vfmap, _ = pearsonr(x_wo_cent_row[~nas], y5_wo_cent_row[~nas])
            r_vimgs_vfmap[j,i] = r_vimg_vfmap

            # 90 rotated images rotated res
            y6 = rfmaps_90rotated_z[j,:]
            y6_wo_cent_pix = torch.masked_select(torch.from_numpy(y6),tensor_wo_center_pix)
            y6_wo_cent_pix = y6_wo_cent_pix.numpy()
            nas = np.logical_or(np.isnan(x_wo_cent_pix),np.isnan(y6_wo_cent_pix))
            r_r90img_rfmap, _ = pearsonr(x_wo_cent_pix[~nas], y6_wo_cent_pix[~nas])
            r_r90imgs_rfmap[j,i] = r_r90img_rfmap

# data dictionary
data_dict = {'r_himgs':r_himgs,
     'r_vimgs':r_vimgs,
     'r_r90imgs':r_r90imgs,
     'r_himgs_hfmap':r_himgs_hfmap,
     'r_vimgs_vfmap':r_vimgs_vfmap,
     'r_r90imgs_rfmap':r_r90imgs_rfmap,
    }

# save the correlation file
with open(figname+'_corr_invaraince_euqivaraince_zscored_'+cond+'_'+str(num_imgs)+'.csv','wb') as fp:
    pickle.dump(data_dict,fp)