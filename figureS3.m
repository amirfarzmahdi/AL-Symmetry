% figure S3: mirror symmetric viewpoint tuning index for pool5 and fc6
% layers
% last update: October 27 2022
% Amirhossein Farzmahdi

clear
close all
clc

% add path
addpath(genpath('functions'))
images_dir = 'mat_files/';

% load data
load(images_dir+"imgs")
load(images_dir+"bw_masks")

% settings
nexemplar = 25;
nview = 9;
ncategory = 9;
nobj = ncategory * nexemplar;
img_size = [227, 227]; % input image size

% matches the mean luminance and contrast of a set of images
lumMatch_imgs = reshape(lumMatch(cm_imgs(:),cm_bw_masks(:)),[nobj, nview]);

% model setting
net = alexnet;
layers = [16,18]; % [16, 18] % select the last convolutional layer and first fully connected layer
layer_name = {net.Layers(:).Name};
RDMs = cell(nobj,length(layers));
msvt_index_values = nan(nobj,length(layers));
num = 0;
for i_layer = layers % number of layers

    num = num+1;
    imgs_res = [];

    % category
    for i_category = 1:ncategory
        object_category = lumMatch_imgs(:,i_category);

        % exemplar
        for i_exemplar = 1:nexemplar

            obj_imgs = object_category(i_exemplar:nexemplar:nview*nexemplar);

            obj_res = [];
            for i_view = 1:nview
                img = obj_imgs{i_view};

                if ~isa(img,'uint8');img = im2uint8(img);end

                % trained network
                imgRGB = cat(3,img,img,img);
                img_ = single(imgRGB) ; % note: 0-255 range
                img_ = imresize(img_, net.Layers(1).InputSize(1:2)) ;
                res = activations(net,img_,i_layer,'ExecutionEnvironment','gpu');
                obj_res = [obj_res squeeze(double(res(:)))];
            end

            imgs_res = [imgs_res, obj_res];
        end
    end

    disp(['---> layer: ' layer_name{i_layer} ' finished'])

    % z-score activation maps
    imgs_res_zscored = zscore(imgs_res,2);

    for i_obj = 1:nobj
        % compute dissimilarity matrix
        obj_res_zscored = imgs_res_zscored(:,1 + ((i_obj - 1) * nview) : i_obj * nview);

        if i_layer == layers(1) % pool5
            tmp = reshape(obj_res_zscored,[size(res),nview]);
            obj_res_zscored = squeeze(mean(tmp,[1,2],'omitnan')); % global average pooling
        end

        % computing RDM
        RDM = 1-corr(obj_res_zscored,'rows','complete');
        RDMs{i_obj,num} = RDM;

        msvt_index_values(i_obj,num) = msvt_index(RDM);

        disp(['---> image ' num2str(i_obj) ' finished'])
    end
end

% figure settings
figure;
cmap = [166,206,227;
    31,120,180;
    178,223,138;
    51,160,44;
    253,191,111;
    255,127,0;
    202,178,214;
    251,154,153;
    227,26,28
    ]/255;

set(gcf,'color',[1 1 1]);
ax = axes;
x = linspace(-1,1);
y = linspace(-1,1);

for i_obj = 1:ncategory
    obj_idx = ((i_obj - 1) * nexemplar) + 1: i_obj * nexemplar;
    scatter(msvt_index_values(obj_idx,1),msvt_index_values(obj_idx,2),30,'MarkerEdgeColor',[1,1,1],...
        'MarkerFaceColor',cmap(i_obj,:),...
        'LineWidth',0.5,'MarkerFaceAlpha',0.7,'MarkerEdgeAlpha',1)
    hold on
end

axis square;box off
hold on
plot(x,y,'Color',[0.5 0.5 0.5],'LineStyle','--','LineWidth',1);

xticks([-1,-0.5,0,0.5,1]);
xticklabels([-1,-0.5,0,0.5,1])
yticks([-1,-0.5,0,0.5,1]);
yticklabels([-1,-0.5,0,0.5,1])

ax.TickDir = 'out';
ax.TickLength = [0.005 0.005];
ax.FontName = 'arial';
ax.FontSize = 6;

text(-0.95,0.9,'n = 225')

xlabel([{'Mirror-symmetric viewpoint tuning'};{'following GAP applied on pool5'}])
ylabel([{'Mirror-symmetric viewpoint tuning'};{'following fc6'}])
