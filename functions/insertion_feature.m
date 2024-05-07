% shuffle local features function
% October 26 2022
% Amirhossein Farzmahdi

function [insertion_images, fc6_selected_units_ins_res_image] = insertion_feature(nview,selected_percent,img_size,bckg,sum_maps,images,net,layer,unit)

for i_view = 1:nview
    one_pixel = find(sum_maps{i_view}(:));
    [~, sort_ind] = sort(abs(sum_maps{i_view}(one_pixel)),'descend','MissingPlacement','last');
    numpixels = round(linspace(0,length(sort_ind),100));
    selected_ind = one_pixel(sort_ind(1: numpixels(selected_percent)));
    
    % insertion
    img = images{i_view};
    ins_img = bckg * ones(img_size);
    tmp = im2double(img);
    ins_img(selected_ind) = tmp(selected_ind);
    imgRGB = cat(3,ins_img,ins_img,ins_img);
    img_ = single(im2uint8(imgRGB)); % note: 0-255 range
    % insertion image
    insertion_images{i_view} = img_;
    
    % fc6 response
    fc6_selected_units_ins_res = activations(net,img_,net.Layers(layer).Name,'Acceleration','mex');
    fc6_selected_units_ins_res_image(:,i_view) = squeeze(fc6_selected_units_ins_res(1,1,unit));
end
