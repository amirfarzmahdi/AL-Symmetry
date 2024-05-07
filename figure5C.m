% figure 5C: insertion and deletion thresholds
% last update: October 26 2022
% Amirhossein Farzmahdi


clear
close all
clc


% add path
addpath(genpath('functions'))
images_dir = 'mat_files/';
result_dir = 'results/';

% loading data
load (images_dir+"names")
load(images_dir+"imgs")
load(images_dir+"bw_masks")
load(images_dir+'figure2B_mat_file.mat','msvt_index_values')
load(images_dir+"5000_masks.mat")

% organize dataset
nexemplar = 25;
nview = 9;
ncategory = 9;
img_size = [227, 227];
nobj = ncategory * nexemplar;

dcmap = flipud(cbrewer('div', 'RdYlBu', 256,'PCHIP'));
icmap = cbrewer('div', 'RdYlBu', 256,'PCHIP');

% matches the mean luminance and contrast of a set of images
imgs_lumMatch = reshape(lumMatch(cm_imgs(:),cm_bw_masks(:)),[nobj, nview]);
net = alexnet;

% select layer & mask setting
fc6_layer_idx = 17;
template = abs(linspace(-90,90,nview))';
n_masks = 5000;
bckg = 0.5020;
fc6_masks_res = [];
percent = [10 20 30 40 50 60 70 80 90];
orig_ins_corr = nan(length(percent),nobj);
orig_del_corr = nan(length(percent),nobj);

selected_units_res = nan(nview,nobj);
s = nan(nobj,1);

% select one unit per object
num = 0;
for i = 1:ncategory
    
    object_category = imgs_lumMatch(:,i);
    bw_category = cm_bw_masks(:,i);
    
    % exemplar
    sample = [];
    for i_exemplar = 1:nexemplar
        num = num +1;
        sample = object_category(i_exemplar:nexemplar:nview*nexemplar);
        bw_sample = bw_category(i_exemplar:nexemplar:nview*nexemplar);
        
        fc6_res = [];
        for i_view = 1:length(sample)
            bw_sample{i_view} = imdilate(bw_sample{i_view}, strel('disk',5));
            img = sample{i_view};
            if ~isa(img,'uint8')
                img = im2uint8(img);
            end
            
            % obtain and preprocess an image
            imgRGB = cat(3,img,img,img);
            img_ = single(imgRGB) ; % note: 0-255 range
            img_ = imresize(img_, net.Layers(1).InputSize(1:2));
            % alexnet feature vectors
            res = activations(net,img_,net.Layers(fc6_layer_idx).Name,'ExecutionEnvironment','gpu');
            fc6_res = [fc6_res squeeze(res(:))];
        end
        
        % load masks data
        load(['Alexnet_fc6_ch_masks_res_' names{num}(1:end-3) '_' num2str(i_exemplar)])
        
        % find channel with mirror-symmetry selectivity
        template_corr = nan(size(fc6_res,1),1);
        sign_corr = nan(size(fc6_res,1),1);
        reflection_corr = nan(size(fc6_res,1),1);
        
        for i_unit = 1:size(fc6_res,1)
            tmp = fc6_res(i_unit,:)';
            template_corr(i_unit,1) = corr(tmp,template,'type','Pearson');
            sign_corr(i_unit,1) = sign(template_corr(i_unit,1));
            reflection_corr(i_unit,1) = corr(tmp,flipud(tmp),'type','Pearson');
        end
        
        ind_sign_p = find(sign_corr == 1);
        [sorted_corr_p,sorted_ind_p] = sort(reflection_corr(ind_sign_p,1),'descend','MissingPlacement','last');
        sort_ind = ind_sign_p(sorted_ind_p);
        
        % select the first sym unit
        selected_unit = sort_ind(1);
        selected_unit_res = fc6_res(selected_unit,:);
        
        fc6_unit_ins_res = single(zeros(nview,length(percent)));
        fc6_unit_del_res = single(zeros(nview,length(percent)));
        
        for i_view = 1:nview
            img = sample{i_view};
            
            % weighted average of RISE masks
            maps = bsxfun(@times,permute(masks(1:n_masks,:,:),[2 3 1]),reshape(fc6_masks_res{i_view}(selected_unit,:),1,1,numel(fc6_masks_res{i_view}(selected_unit,:))));
            mean_maps = mean(maps,3);
            
            % deletion/insertion
            mmaps = mean(mean_maps(:));
            stdmaps = std(mean_maps(:));
            mean_maps_minus_mean = mean_maps - mmaps;
            mean_maps_minus_mean = mean_maps_minus_mean .* bw_sample{i_view};
            one_pixel = find(mean_maps_minus_mean(:));
            [sort_val, sort_ind] = sort(abs(mean_maps_minus_mean(one_pixel)),'descend','MissingPlacement','last');
            fc6_unit_rise_map{i_view} = [one_pixel sort_ind];
            
            % fraction of pixels
            del_imgs = [];
            ins_imgs = [];
            numpixels = round(linspace(0,length(sort_ind),100));
            n = 0;
            for p = percent
                n = n + 1;
                selected_ind = one_pixel(sort_ind(1: numpixels(p)));
                
                % deletion
                del_img = im2double(img);
                del_img(selected_ind) = bckg;
                imgRGB = cat(3,del_img,del_img,del_img);
                img_ = single(im2uint8(imgRGB)); % note: 0-255 range
                img_ = imresize(img_, net.Layers(1).InputSize(1:2)) ;
                fc6_del_res = activations(net,img_,net.Layers(fc6_layer_idx).Name,'ExecutionEnvironment','gpu');
                fc6_unit_del_res(i_view,n) = fc6_del_res(selected_unit);
                del_imgs = [del_imgs im2uint8(del_img(:))];
                
                % insertion
                ins_img = bckg * ones(img_size);
                tmp = im2double(img);
                ins_img(selected_ind) = tmp(selected_ind);
                imgRGB = cat(3,ins_img,ins_img,ins_img);
                img_ = single(im2uint8(imgRGB)); % note: 0-255 range
                img_ = imresize(img_, net.Layers(1).InputSize(1:2));
                fc6_ins_res = activations(net,img_,net.Layers(fc6_layer_idx).Name,'ExecutionEnvironment','gpu');
                
                fc6_unit_ins_res(i_view,n) = fc6_ins_res(selected_unit);
                ins_imgs = [ins_imgs im2uint8(ins_img(:))];
            end
            display(['view: ',num2str(i_view)])
        end
        
        orig_res = selected_unit_res';
        ins_corr = nan(1,length(percent));
        del_corr = nan(1,length(percent));
        for i_percent = 1:length(percent)
            % corr orig with insertion
            ins_res = squeeze(fc6_unit_ins_res(:,i_percent));
            ins_corr(1,i_percent) = corr(ins_res,orig_res); % orig_res flipud(ins_res)
            
            % corr orig with deletion
            del_res = squeeze(fc6_unit_del_res(:,i_percent));
            del_corr(1,i_percent) = corr(del_res,orig_res); % orig_res flipud(del_res)
        end
        
        % clear mask results 
        clear maps fc6_masks_res
        
        % corr. between insertion/deletion & original response
        orig_ins_corr(:,num) = ins_corr;
        orig_del_corr(:,num) = del_corr;
        selected_channels(num,1) = selected_unit;
        selected_units_res(:,num) = orig_res;
        disp(['image: ', num2str(num), ' finished']);
    end
end

% plot correlation between original and deletion/insertion
orig_del_corr = flipud(orig_del_corr);

std_ins = std(orig_ins_corr,[],2);
std_del = std(orig_del_corr,[],2);

median_ins = median(orig_ins_corr,2);
median_del = median(orig_del_corr,2);

npercent = length(percent);
jitter_ = 0.3;

difference_corr = orig_ins_corr - orig_del_corr;
median_difference = median(difference_corr,2);
std_difference = std(difference_corr,[],2)/sqrt(nobj);

figure;
ax = axes;
hold on

yyaxis left

scatter(1:npercent,median_ins,15,dcmap(1,:),'filled','MarkerEdgeColor',dcmap(1,:),'MarkerEdgeAlpha',1)
errorbar(1:npercent,median_ins,std_ins,'Color',dcmap(1,:),'Marker','none','LineWidth',1,'LineStyle','none','AlignVertexCenters','on','CapSize',0);

scatter(jitter_+(1:npercent),median_del,15,icmap(1,:),'filled','MarkerEdgeColor',icmap(1,:),'MarkerEdgeAlpha',1)
errorbar(jitter_+(1:npercent),median_del,std_ins,'Color',icmap(1,:),'Marker','none','LineWidth',1,'LineStyle','none','AlignVertexCenters','on','CapSize',0);

xtickangle(45)
xticks([1 1.3 2 2.3 3 3.3 4 4.3 5 5.3 6 6.3 7 7.3 8 8.3 9 9.3])
xticklabels([{'10% insertion'};{'90% deletion'};{'20% insertion'};{'80% deletion'};{'30% insertion'};...
    {'70% deletion'};{'40% insertion'};{'60% deletion'};{'50% insertion'};{'50% deletion'};...
    {'60% insertion'};{'40% deletion'};{'70% insertion'};{'30% deletion'};{'80% insertion'};...
    {'20% deletion'};{'90% insertion'};{'10% deletion'}])

ylabel({'Correlation with original'; 'viewpoint tuning profile'});
yticks([-0.5 0 0.5 1]);
yticklabels([{'-0.5'};{'0'};{'0.5'};{'1'}]);
ylim([-0.5 1.2]);
xlim([0.5 9.5]);

yyaxis right
bar((jitter_/2)+(1:npercent),median_difference,0.15,'FaceColor',[0.6 0.6 0.6],'EdgeColor',[0.6 0.6 0.6],'LineWidth',0.5,'FaceAlpha',0.4,'EdgeAlpha',0.4)
errorbar((jitter_/2)+(1:npercent),median_difference,std_difference,'Color',[0.6 0.6 0.6],'LineWidth',0.7,'LineStyle','none','AlignVertexCenters','on','CapSize',0);

ylabel({'Difference between insertion';  'and deletion correlations'});
yticks([0 0.5 1]);
yticklabels([{'0'};{'0.5'};{'1'}]);
ylim([0 1.2])

ax.TickDir='out';ax.TickLength=[0.005 0.005];

ax.YAxis(1).Color = [0.15 0.15 0.15];
ax.YAxis(2).Color = [0.6 0.6 0.6];