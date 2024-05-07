% figure 5A: example of RISE method
% last update: October 26 2022
% Amirhossein Farzmahdi

clear
close all
clc

% fixed random seed for regenerating same result
seed = 42; rng(seed)

% add path
addpath(genpath('functions'))
images_dir = 'mat_files/';
result_dir = 'results/';

% loading data
load(images_dir+"5000_masks.mat")
load(images_dir+"names.mat")

% object 5A
load([result_dir,'/mat_files/car22.mat'])
% object 5B
% load([result_dir,'/mat_files/face1.mat'])

% organize dataset
nview = 9;
img_size = [227, 227];
nexemplar = 25;


% unit setting
npercent = 25;
unit = 3995; % 5A: 3995, 5B: 3183
figure_name = 'figure5A'; % figure5B
catg = 1; % 5A: 1, 5B: 3
exemplar = 22; % 5A: 22, 5B: 1
nimg = (catg - 1) * nexemplar + exemplar;

% load network
net = alexnet;
tmp_net = net.saveobj;
mean_train_image = tmp_net.Layers(1).AverageImage;
mean_ch1 = round(mean(squeeze(mean_train_image(:,:,1)),'all'));
mean_ch2 = round(mean(squeeze(mean_train_image(:,:,2)),'all'));
mean_ch3 = round(mean(squeeze(mean_train_image(:,:,3)),'all'));
mean_train_image_ = cat(3,mean_ch1 .* ones(img_size),mean_ch2 .* ones(img_size),...
    mean_ch3 .* ones(img_size));
tmp_net.Layers(1).AverageImage = mean_train_image_;
net = net.loadobj(tmp_net);

% main settings
layer = 17; % fc6 layer
template = abs(linspace(-90,90,nview))';
n_mask = 5000;
n_shuffle = 1000;
max_percent = 100;
bckg = 0.5020;
dcmap = flipud(cbrewer('div', 'RdYlBu', 256,'PCHIP'));
icmap = cbrewer('div', 'RdYlBu', 256,'PCHIP');
n_repeat = 1000;
[rise_mean_maps_views, insertion_images] = deal(cell(1,nview));
fc6_unit_del_res_views_percents = nan(1,nview,max_percent);
fc6_unit_ins_res_views_percents = nan(1,nview,max_percent);
fc6_unit_del_views = cell(1,nview);
fc6_unit_ins_views = cell(1,nview);

% load masks data
load([result_dir,'/mat_files/Alexnet_fc6_ch_masks_res_' names{nimg}(1:end-3) '_' num2str(exemplar)])

for i_view = 1:nview
    
    % dilating the binary mask to cover the surrounding pixels
    bw_img = imdilate(bw_sample{i_view}, strel('disk',5));
    
    % weighted average of RISE masks
    maps = bsxfun(@times,permute(masks(1:n_mask,:,:),[2, 3, 1]),reshape(fc6_masks_res{i_view}(unit,:),1,1,numel(fc6_masks_res{i_view}(unit,:))));
    mean_maps = mean(maps,3);
    
    % deletion/insertion
    mmaps = mean(mean_maps(:));
    mean_maps_minus_mean = mean_maps - mmaps;
    mean_maps_minus_mean = mean_maps_minus_mean .* bw_img;
    
    rise_mean_maps_views{1,i_view} = mean_maps_minus_mean;
    
    one_pixel = find(mean_maps_minus_mean(:));
    [~, sort_ind] = sort(abs(mean_maps_minus_mean(one_pixel)),'descend','MissingPlacement','last');
    
    img = sample{i_view};
    if ~isa(img,'uint8')
        img = im2uint8(img);
    end
    
    % fraction of pixels
    n = 0;
    deletion_imgs = nan([img_size,max_percent]);
    insertion_imgs = nan([img_size,max_percent]);
    numpixels = round(linspace(0,length(sort_ind),max_percent));
    
    for p = 1:max_percent
        n = n+1;
        selected_pixel = one_pixel(sort_ind(1: numpixels(p)));
        
        % response to the deletion image
        delimg = im2double(img);
        delimg(selected_pixel) = bckg;
        imgRGB = cat(3,delimg,delimg,delimg);
        img_ = single(im2uint8(imgRGB)); % note: 0-255 range
        img_ = imresize(img_, net.Layers(1).InputSize(1:2)) ;
        fc6_del_res = activations(net,img_,net.Layers(layer).Name,'ExecutionEnvironment','gpu');
        fc6_unit_del_res_views_percents(1,i_view,n) = fc6_del_res(unit);
        deletion_imgs(:,:,n) = im2uint8(delimg);
        
        % response to the insertion image
        insimg = bckg * ones(img_size);
        tmp = im2double(img);
        insimg(selected_pixel) = tmp(selected_pixel);
        imgRGB = cat(3,insimg,insimg,insimg);
        img_ = single(im2uint8(imgRGB)); % note: 0-255 range
        img_ = imresize(img_, net.Layers(1).InputSize(1:2)) ;
        fc6_ins_res = activations(net,img_,net.Layers(layer).Name,'ExecutionEnvironment','gpu');
        % insertion image
        if p == npercent
            insertion_images{i_view} = img_;
            pixels{i_view} = selected_pixel;
        end
        
        fc6_unit_ins_res_views_percents(1,i_view,n) = fc6_ins_res(unit);
        insertion_imgs(:,:,n) = im2uint8(insimg);
    end
    fc6_unit_del_views{1,i_view} = deletion_imgs;
    fc6_unit_ins_views{1,i_view} = insertion_imgs;
end

[shuffeled_example,fc6_unit_shuffeled_res_views] = shuffle_feature(nview,n_repeat,n_shuffle,img_size,bckg,insertion_images,pixels,net,layer,unit);

% panel: i
hi = figure;
set(gcf, 'Position', [1 1 800 200],'color',[1 1 1]);
h_i = tight_subplot(1,nview,[.005 .005],[.01 .01],[.01 .075]);

% panel: ii
figure;
set(gcf, 'Position', [1 1 800 200],'color',[1 1 1]);
h_ii = tight_subplot(1,nview,[.005 .005],[.01 .01],[.01 .05]);

% panel: iii
figure;
set(gcf, 'Position', [1 1 800 200],'color',[1 1 1]);
h_iii = tight_subplot(1,nview,[.005 .005],[.01 .01],[.01 .05]);

% panel: iv
figure;
set(gcf, 'Position', [1 1 800 200],'color',[1 1 1]);
h_iv = tight_subplot(1,nview,[.005 .005],[.01 .01],[.01 .05]);

max_val_rise = cellfun(@(x) max(x(:)), rise_mean_maps_views(1,:), 'UniformOutput', false);
max_val_rise = max(cell2mat(max_val_rise));
min_val_rise = cellfun(@(x) min(x(:)), rise_mean_maps_views(1,:), 'UniformOutput', false);
min_val_rise = min(cell2mat(min_val_rise));

for i_view = 1:nview
    
    
    % plot panel: i
    axes(h_i(i_view));
    imoverlay_manual(double(sample{i_view})/255,(rise_mean_maps_views{1,i_view}-mean(rise_mean_maps_views{1,i_view}(:))),[],[],[],[],h_i(i_view));
    box off;axis off;axis square
    colormap(flipud(cbrewer('div', 'Spectral', 256, 'PCHIP')));
    h_i(i_view).CLim = [min_val_rise max_val_rise];
    
    % plot panel: ii
    axes(h_ii(i_view));
    imshow(double(squeeze(fc6_unit_del_views{1,i_view}(:,:,npercent)))/255);
    axis off;box off;axis square;colormap(gray)
    
    % plot panel: iii
    axes(h_iii(i_view));
    imshow(double(squeeze(fc6_unit_ins_views{1,i_view}(:,:,npercent)))/255);
    axis off;box off; axis square;colormap(gray)
    
    % plot panel: iv
    axes(h_iv(i_view));
    imshow(double(squeeze(shuffeled_example{i_view}(:,:,1)))/255);
    axis off;box off; axis square;colormap(gray)
    
end

if i_view == 9 && ~isempty(hi.CurrentAxes) %last panel
        hold on
        axes(h_i(i_view));
        
        cr = colorbar;
        if max_val_rise > 0 && max_val_rise < 1
            cr.Limits = [round(min_val_rise*10)/10 round(max_val_rise*10)/10];
            cr.Ticks = [round(min_val_rise*10)/10 round(max_val_rise*10)/10];
            cr.TickLabels = [round(min_val_rise*10)/10 round(max_val_rise*10)/10];
        else
            cr.Limits = round([min_val_rise max_val_rise]);
            cr.Ticks = round([min_val_rise max_val_rise]);
            cr.TickLabels = round([min_val_rise max_val_rise]);
        end
        cr.Box = 'off';
        cr.Position = cr.Position + 1e-10;
        cr.Position(1) = cr.Position(1)+0.05;
        cr.Position(4) = cr.Position(4)+0.25;
        cr.Position(3) = cr.Position(3)+0.0055;
        cr.Position(2) = cr.Position(2)-0.14;
        cr.FontSize = 6;
        cr.FontWeight = 'normal';
        cr.Box = 'off';
        cr.FontName = 'arial';
        cr.TickDirection = 'out';
        cr.TickLength = 0.001;
end
    
% plot panel: v
h_v = figure;
ax = axes;
hold on

% shuffled response
mean_ins_rand_pos = mean(squeeze(fc6_unit_shuffeled_res_views),2);
std_ins_rand_pos = std(squeeze(fc6_unit_shuffeled_res_views),[],2);%/sqrt(n_shuffle);

max_ = max(fc6_unit_ins_res_views_percents(1,:,:),[],'all');
max_ = (ceil(max_) * 10)/10;
min_ = min(fc6_unit_ins_res_views_percents(1,:,:),[],'all');
min_ = (floor(min_) * 10)/10;

% original
plot(squeeze(fc6_unit_del_res_views_percents(1,:,1)),'LineWidth',1,'Color',[0.6 0.6 0.6],'Marker','o',...
    'MarkerFaceColor',[0.6 0.6 0.6],'MarkerSize',3,'MarkerEdgeColor',[0.6 0.6 0.6]);

% deletion
plot(squeeze(fc6_unit_del_res_views_percents(1,:,npercent)),'LineWidth',1,'Color',icmap(1,:),'Marker','o',...
    'MarkerFaceColor',icmap(1,:),'MarkerSize',3,'MarkerEdgeColor',icmap(1,:));

% insertion
plot(squeeze(fc6_unit_ins_res_views_percents(1,:,npercent)),'LineWidth',1,'Color',dcmap(1,:),'Marker','o','MarkerFaceColor',dcmap(1,:),...
    'MarkerSize',3,'MarkerEdgeColor',dcmap(1,:));

% shuffled
plot(mean_ins_rand_pos,'LineWidth',1,'LineStyle','-','Color',[151 203 255]/255,'Marker','o',...
    'MarkerFaceColor',[151 203 255]/255,'MarkerSize',3,'MarkerEdgeColor',[151 203 255]/255);
errorbar(1:nview,mean_ins_rand_pos,std_ins_rand_pos,'Color',[151 203 255]/255,'LineWidth',1.5,'LineStyle','none','AlignVertexCenters','on','CapSize',0);

line(linspace(0.5,nview+0.5,nview),fc6_unit_ins_res_views_percents(1,:,1),'LineStyle','--','Color',[0 0 0]);

% panel v, plot setting
opt = [];
opt.XLabel = 'Viewpoints'; % xlabel
opt.YLabel = 'Channel response (a.u.)'; %ylabel
opt.XTick = 1:nview;
opt.YLim = [min_ max_];
opt.XLim = [0.85 nview+0.15];
opt.ShowBox = 'off';
opt.Colors = [[0.6 0.6 0.6];icmap(1,:); dcmap(1,:); [151 203 255]/255; [151 203 255]/255; [0 0 0]];
opt.LineWidth = 1 * ones(1,56);
opt.AxisLineWidth = 0.8;
opt.LineStyle = {'-', '-', '-','-','-','--'}; % three line styles
opt.Markers = repelem({'o'},1,4);
opt.Legend = {'original','deletion','insertion','shuffled'}; % legends
opt.XMinorTick = 'off';
opt.YMinorTick = 'off';
opt.TickDir = 'out';
opt.TickLength = [0.001 0.001];
opt.LegendLoc = 'north';
opt.FontSize = 6;
opt.FontName = 'Arial';
opt.BoxDim = [1.4 1.5];% maximum: 8.75 7.5

setPlotProp(opt);

ax.FontName = 'arial';
xtick = linspace(1,nview,nview);
xticklabel = {'-90°','-67.5°','-45°','-22.5°','0°','+22.5°','+45°','+67.5°','+90°'};
set(gca,'xtick',xtick,'xticklabel',xticklabel,...
    'FontSize',6,'FontName','arial','FontWeight','normal');
set(gca,'ytick',[])
xtickangle(ax,45);

legend('off');