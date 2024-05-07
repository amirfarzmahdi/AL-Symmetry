% generate small size images
% last update: October 25 2022
% Amirhossein Farzmahdi

clear
close all
clc

% add path
addpath(genpath('functions'))
images_dir = 'mat_files/';

category_names = [{'car'};{'boat'};{'face'};{'chair'};{'airplane'};{'tool'};{'animal'};{'fruit'};{'flower'}];
load (images_dir+"names")
load(images_dir+"cm_imgs");
load(images_dir+"cm_bw_masks");

nexemplar = 25;
nview = 9;
ncategory = 9;
nobj = nview * nexemplar;
pixel_size = 32;
img_dir = 'small_images/';

% matches the mean luminance and contrast of a set of images
imgs = reshape(lumMatch(cm_imgs(:),cm_bw_masks(:)),[nobj, nview]);

for i_category = 1:ncategory
    mkdir([img_dir category_names{i_category}]);
    fid = fopen([img_dir '/' category_names{i_category} '/' category_names{i_category} '.txt'],'w');
    n = 0;
    for i_obj = 1:nobj
        n = n + 1;
        name = category_names{i_category};
        
        tmp = imgs{n,i_category};
        tmp = imresize(tmp,[pixel_size pixel_size]);
        tmp = cat(3,tmp,tmp,tmp);
        
        if 0 < i_obj && i_obj < 10
            exemplar_name = [name '00'];
        elseif 10 <= i_obj && i_obj < 100
            exemplar_name = [name '0'];
        end
        imwrite(tmp,[img_dir '/' category_names{i_category} '/' exemplar_name num2str(n) '.png']);
        fprintf(fid, [exemplar_name num2str(n) '\n']);
    end
end