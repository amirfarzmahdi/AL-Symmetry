% shuffle local features function
% October 26 2022
% Amirhossein Farzmahdi

function [imgs_shuffled,fc6_unit_shuffeled_views_percents] = shuffle_feature(nview,n_repeat,n_shuffle,img_size,bckg,images,pixels,net,layer,unit)

for i_view = 1:nview
    
    % insertion image
    img_ = images{i_view};
       
    % shuffle the features
    tmp_bw = zeros(img_size);
    tmp_bw(pixels{i_view}) = 1;
    tmp_bw = imbinarize(tmp_bw);
    tmp_img = double(img_)/255;
    info = regionprops(tmp_bw,'Boundingbox','Image');
    rpos1 = [];
    rpos2 = [];
    % sort patches according to their size
    patch_area = nan(length(info),1);
    for i_patch = 1:length(info)
        patch_area(i_patch,1) = size(info(i_patch).Image,1) * size(info(i_patch).Image,2);
    end
    [~,sort_patch_ind] = sort(patch_area,'descend');

            
    for rn = 1:n_shuffle
        tmp_rand = bckg * ones(img_size);
        ind = zeros(img_size);
        flag1 = 0;
        for k = 1 : length(info)
            BB = info(sort_patch_ind(k)).BoundingBox;
            BB_ = fix(BB);
            if ~isempty(find(BB_ == 0,1))
                BB_(find(BB_ == 0,1)) = 1;
            end
            patch_ = tmp_img(BB_(2):BB_(2)+BB_(4)-1,BB_(1):BB_(1)+BB_(3)-1);
            
            % remove square shape from the selected patch
            tmp_patch = patch_ - bckg;
            tmp_patch = tmp_patch .* info(sort_patch_ind(k)).Image;
            patch_ = tmp_patch + bckg;
            bpatch_ = patch_;
            bpatch_(patch_ ~= bckg) = nan;
            bpatch_ = bpatch_ - bckg;
            
            % random feature positions
            flag = 0;
            count = 0;
            while flag == 0
                count = count + 1;
                [rpos1, rpos2] = ind2sub(size(ind),randsample(find(~isnan(ind(:))),1));
                
                if rpos1 > 0 && rpos2 > 0 && ...
                        rpos1 + size(patch_,1) < img_size(1) && rpos2 + size(patch_,2) < img_size(2)
                    
                    if sum(isnan(ind(rpos1:rpos1+BB_(4)-1,rpos2:rpos2+BB_(3)-1)),'all') == 0
                        flag = 1;
                        ind(rpos1:rpos1+BB_(4)-1,rpos2:rpos2+BB_(3)-1) = bpatch_;
                        rpos1_old = rpos1;
                        rpos2_old = rpos2;
                    end
                end
                
                if count > n_repeat % if run number higher than threshold exist the loop and use the previous shuffled image
                    flag1 = 1;
                    nrepeat = nrepeat + 1;
                    break
                end
            end
            if flag1 ~= 1
                tmp_rand(rpos1:rpos1+BB_(4)-1,rpos2:rpos2+BB_(3)-1) = patch_;
                ind(rpos1:rpos1+BB_(4)-1,rpos2:rpos2+BB_(3)-1) = bpatch_;
            else
                break
            end
        end
        
        if flag1 == 1
            img_sh = img_;
        else
            imgRGB = cat(3,tmp_rand,tmp_rand,tmp_rand);
            img_sh = single(im2uint8(imgRGB));
            img_sh = imresize(img_sh, net.Layers(1).InputSize(1:2));
        end
        
        fc6_unit_shuffeled_res = activations(net,img_sh,net.Layers(layer).Name,'Acceleration','mex');
        fc6_unit_shuffeled_views_percents(:,i_view,rn) = squeeze(fc6_unit_shuffeled_res(1,1,unit));
        
        imgs_shuffled{i_view}(:,:,rn) = im2uint8(tmp_rand);
    end
    disp(['view  ' num2str(i_view) '---> is finished'])
end