% generate masked images
% last update: October 26 2022
% Amirhossein Farzmahdi

clear
close all
clc

% add path
addpath(genpath('functions'))
images_dir = 'mat_files/';

% loading data
load (images_dir+"names.mat")
load(images_dir+"imgs.mat")
load(images_dir+"bw_masks.mat")
load(images_dir+"5000_masks.mat")

% setting
nexemplar = 25;
nview = 9;
ncategory = 9;
img_size = [227, 227];
nobj = ncategory * nexemplar;
fc6_layer_idx = 17;
nmask = 5000;
bckg = 0.5020;

% matches the mean luminance and contrast of a set of images
imgs_lumMatch = reshape(lumMatch(cm_imgs(:),cm_bw_masks(:)),[nobj, nview]);

% load network
net = alexnet;

num = 0;
for i_category = 1:ncategory
    object_category = imgs_lumMatch(:,i_category);
    
    % exemplar
    sample = [];
    for i_exemplar = 1:nexemplar
        num = num + 1;
        exemplar = str2double(names{num}(end - 2 : end));
        sample = object_category(i_exemplar : nexemplar : nview * nexemplar);
        
        fc6_res = [];
        for i_view = 1:length(sample)
            img = sample{i_view};
            if ~isa(img,'uint8')
                img = im2uint8(img);
            end
            
            % obtain and preprocess an image
            imgRGB = cat(3,img,img,img);
            img_ = single(imgRGB) ; % note: 0-255 range
            img_ = imresize(img_, net.Layers(1).InputSize(1:2));
            res = activations(net,img_,net.Layers(fc6_layer_idx).Name,'ExecutionEnvironment','gpu');
            fc6_res = [fc6_res squeeze(res(:))];
            
            % obtain mask output
            mask = [];
            masked_img = [];
            masked_imgs_fc6_res = [];
            img = im2double(img) - bckg;
            tic
            for i_mask = 1:nmask
                mask = squeeze(masks(i_mask,:,:));
                masked_img =  mask .* img;
                masked_img = masked_img + bckg;
                imgRGB = cat(3,masked_img,masked_img,masked_img);
                masked_img_ = single(im2uint8(imgRGB)); % note: 0-255 range
                masked_img_ = imresize(masked_img_, net.Layers(1).InputSize(1:2)) ;
                masked_img_fc6_res = activations(net,masked_img_,net.Layers(fc6_layer_idx).Name,'ExecutionEnvironment','gpu');
                masked_imgs_fc6_res = [masked_imgs_fc6_res squeeze(masked_img_fc6_res)];
            end
            toc
            fc6_masks_res{i_view} = masked_imgs_fc6_res;
        end
        save(['new/' 'Alexnet_fc6_ch_masks_res_' names{num}(1:end-3) '_' num2str(exemplar)],...
            'fc6_masks_res','-v7.3')
        
        display(names{num})
    end
end
