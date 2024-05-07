% figure 5D: suffeled vs insertion mirror-symmetric viewpoint tuning index
% last update: October 28 2022
% Amirhossein Farzmahdi

clear
close all
clc

% fixed random seed for regenerating same result
seed = 42; rng(seed)

% add path
addpath(genpath('functions'))
images_dir = 'mat_files/';

% loading data
load (images_dir+"names")
load(images_dir+"imgs")
load(images_dir+"bw_masks")
load(images_dir+'figure2D_S2_mat_file.mat','msvt_index_values')
load(images_dir+"5000_masks.mat")

% settings
nexemplar = 25;
nview = 9;
ncategory = 9;
img_size = [227, 227];
nobj = nview * nexemplar;
n_shuffle = 1000;
n_repeat = 1000;
layer = 17;
template = abs(linspace(-90,90,nview))';
n_masks = 5000;
bckg = 0.5020;
percentile_level = 95;
selected_percent = 25;
fc6_masks_res = [];

% matches the mean luminance and contrast of a set of images
lumMatch_imgs = reshape(lumMatch(cm_imgs(:),cm_bw_masks(:)),[nobj, nview]);

% load network
net = alexnet;
[selected_units_per_obj, fc6_selected_units_res, fc6_selected_units_shuffle_res_images, ...
    fc6_selected_units_ins_res_images] = deal(cell(nobj,1));

i_obj = 0;
for i_category = 1:ncategory
    object_category = lumMatch_imgs(:,i_category);
    bw_category = cm_bw_masks(:,i_category);
    
    % exemplar
    sample = [];
    for i_exemplar = 1:nexemplar
        
        i_obj = i_obj + 1;
        % load masks data
        load(['Alexnet_fc6_ch_masks_res_' names{i_obj}(1:end-3) '_' num2str(i_exemplar)]);
        sample = object_category(i_exemplar:nexemplar:nview * nexemplar);
        bw_sample = bw_category(i_exemplar:nexemplar:nview * nexemplar);
        
        fc6_res = [];
        for i_view = 1:nview
            bw_sample{i_view} = imdilate(bw_sample{i_view}, strel('disk',5));
            img = sample{i_view};
            % original model response
            imgRGB = cat(3,img,img,img);
            img_ = single(imgRGB) ; % note: 0-255 range
            img_ = imresize(img_, net.Layers(1).InputSize(1:2));
            % alexnet activation maps
            res = activations(net,img_,net.Layers(layer).Name,'Acceleration','mex');
            fc6_res = [fc6_res squeeze(res(:))];
        end
        
        % find units with high mirror-symmetric viewpoint tuning
        corr_refl = nan(size(fc6_res,1),1);
        sign_corr_refl = nan(size(fc6_res,1),1);
        corr_flip = nan(size(fc6_res,1),1);
        for i_unit = 1:size(fc6_res,1)
            tmp = fc6_res(i_unit,:)';
            corr_refl(i_unit,1) = corr(tmp,template,'type','Pearson');
            sign_corr_refl(i_unit,1) = sign(corr_refl(i_unit,1));
            corr_flip(i_unit,1) = corr(tmp,flipud(tmp),'type','Pearson');
        end
        
        ind_sign_p = find(sign_corr_refl == 1);
        [sorted_corr_p,sorted_ind_p] = sort(corr_flip(ind_sign_p,1),'descend','MissingPlacement','last');
        Y = prctile(sorted_corr_p,percentile_level);
        tmp = find(sorted_corr_p > Y);
        selected_unit = ind_sign_p(sorted_ind_p(tmp))';
        
        % fc6 response of selected units to origianl images
        fc6_selected_units_res{i_obj} = fc6_res(selected_unit,:);
        
        n = 0;
        sum_maps = mat2cell(repmat(zeros(img_size),1,9),img_size(1),...
            repmat(img_size(2),1,nview));
        tic
        for unit = selected_unit
            n = n + 1;
            for i_view = 1:nview
                img = sample{i_view};
                
                maps = bsxfun(@times,permute(masks(1:n_masks,:,:),[2 3 1])...
                    ,reshape(fc6_masks_res{i_view}(unit,1:n_masks),1,1,length(1:n_masks)));
                
                mean_maps = mean(maps,3);
                mmaps = mean(mean_maps(:));
                mean_maps_minus_mean = mean_maps - mmaps;
                mean_maps_minus_mean = mean_maps_minus_mean .* bw_sample{i_view};
                
                sum_maps{i_view} = sum_maps{i_view} + mean_maps_minus_mean;
            end
            disp(['unit ' num2str(unit) ': analysis is finished - ' 'remaining units: ' num2str(length(selected_unit) - n)])
        end
        
        % inserted salient pixels
        [insertion_images,fc6_selected_units_ins_res_images{i_obj}] = insertion_feature(nview,selected_percent,img_size,bckg,sum_maps,sample,net,layer,selected_unit);

        % shuffle the salient pixels
        [shuffeled_images,fc6_unit_shuffeled_views_percents] = shuffle_feature(nview,n_repeat,n_shuffle,img_size,bckg,insertion_images,net,layer,selected_unit);
        fc6_selected_units_shuffle_res_images{i_obj} = median(fc6_unit_shuffeled_views_percents,3);
                
        % selected units per object
        selected_units_per_obj{i_obj} = selected_unit;
    end
end