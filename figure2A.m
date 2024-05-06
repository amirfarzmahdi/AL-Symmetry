% figure 2A: measuring sample dissimilarity matrix
% last update: October 24 2022
% Amirhossein Farzmahdi

clear
close all
clc

% add path
addpath(genpath('functions'))
images_dir = 'mat_files/';

% loading data
load (images_dir+"names")
load(images_dir+"imgs")
load(images_dir+"bw_masks")

% setting
nexemplar = 25;
nview = 9;
ncategory = 9;
nobj = ncategory * nexemplar;
objs = [71, 90, 17, 210]; % selected objects: face, chair, car, and flower
nobj_selected = length(objs);

% model setting
net = alexnet;
layers = [1, 7, 11, 13, 15, 18]; % relu layers
nlayer = length(layers);
name_layers = [{'image'};{'conv2'};{'conv3'};{'conv4'};{'conv5'};{'fc6'}];

% matches the mean luminance and contrast of a set of images
imgs_lumMatch = reshape(lumMatch(cm_imgs(:),cm_bw_masks(:)),[nobj, nview]);

RDM_mats = cell(nexemplar * ncategory, length(layers));
corr_vals = zeros(nexemplar * ncategory, length(layers));
imgs = cell(nexemplar, ncategory);

% figure setting
figure
set(gcf,'color',[1 1 1],'Position', [1 1 800 600]);
h = tight_subplot(nobj_selected,nlayer,[.005 .02],[.01 .01],[.01 .01]);
fig_idx = reshape(1:nobj_selected * nlayer,[nlayer,nobj_selected])';
fig_idx = fig_idx(:);
colormap(gray)

num = 0;
for i_layer = 1:length(layers) % number of layers
    imgs_res = [];
    
    % category
    for i_category = 1 : ncategory
        object_category = imgs_lumMatch(:,i_category);
        
        % exemplar
        sample = [];
        for i_exemplar = 1 : nexemplar
            
            sample = object_category(i_exemplar : nexemplar: nview * nexemplar);
            obj_res = [];
            
            for i_view = 1 : nview
                img = sample{i_view};
                
                if ~isa(img,'uint8');img = im2uint8(img);end
                
                % trained network
                imgRGB = cat(3,img,img,img);
                img_ = single(imgRGB) ; % note: 0-255 range
                img_ = imresize(img_, net.Layers(1).InputSize(1:2)) ;
                img_res = activations(net,img_,layers(i_layer),'ExecutionEnvironment','gpu');
                obj_res = [obj_res squeeze(double(img_res(:)))];
                
            end
            imgs_res = [imgs_res, obj_res];
        end
    end
    
    % z-score activation maps
    imgs_res_zscored = zscore(imgs_res);
    
    for i_obj = objs
        num = num + 1;
        % compute dissimilarity matrix
        obj_res_zscored = imgs_res_zscored(:,1 + ((i_obj - 1) * nview) : i_obj * nview);
        RDM = 1 - corr(obj_res_zscored,'rows','complete');
        
        corr_val = msvt_index(RDM);
        
        RDM_scaled = scale01(rankTransform_equalsStayEqual(RDM,1));
        axes(h(fig_idx(num)));
        imshow(RDM_scaled);
        axis off; box off; 
        
        disp(['---> layer: ' name_layers{i_layer}...
            '   ' '---> object ' num2str(names{i_obj}) ...
            '  msvt index: ' num2str(round(corr_val * 100)/100)])
    end
end