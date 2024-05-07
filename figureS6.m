% figure S6: measuring mirror-symmetric viewpoint tuning for Leibo and
% alexnet models
% last update: April 03 2023
% Amirhossein Farzmahdi

clear
close all
clc

% main settings
ncategory = 9;
nview = 9;
nexemplar = 25;
nobj = ncategory * nexemplar;
ncond = 2;
npc = 39;
n = 227;
m = 227;
savefig = 1;

% add path
addpath(genpath('functions'));
images_dir = 'mat_files/';

% loading data
load(images_dir+"imgs")
load(images_dir+"bw_masks")

imgs_lumMatch = reshape(lumMatch(cm_imgs(:),cm_bw_masks(:)),[nobj, nview]);


%specify directories for training and testing images
train_set.imgs   = 'datasets/BFM_illumination_faces';

train_imgs = readAllImages(train_set,[n,m]); %cI is a cell containing all training images
train_imgs_C1_layer = hmax_C1_layer(train_imgs{1});

% our face dataset
test_imgs{1} = imgs_lumMatch(:,3);
test_imgs_C1_layer{1} = hmax_C1_layer(test_imgs{1});

% shift test images by offset
offsets = [15, 15]; % delta_y delta_x
for i = 1:length(test_imgs{1})
    test_imgs{2}{i,1} = circshift(imgs_lumMatch{i,3},offsets);
end
test_imgs_C1_layer{2} = hmax_C1_layer(test_imgs{2});

% AlexNet model setting
trained_net = alexnet; % trained network
layers = [1, 3, 5, 7, 9, 11, 13, 15, 16, 18, 21];
name_layers = [{'image'};{'conv1'};{'pool1'};{'conv2'};{'pool2'};...
    {'conv3'};{'conv4'};{'conv5'};{'pool5'};{'fc6'};{'fc7'}];

% figure settings
figure('units','inch','position',[0,0,6,2],'color',[1 1 1]);
h = tight_subplot(1, 3, [0.01,0.05], [0.09,0.15], [0.1,0.01]);
rows = [{'pixel'};{'c1'};{'alexnet'}];
nid = 40;
ntrview = 39;
xticklabel = [{'centered'};{'shifted'}];
colors = [27,158,119;117,112,179]/255;
titles = [{'PCA (pixel)'};{'PCA (C1 layer)'};{'AlexNet (fc6)'}];

for i_row = 1:3 % pixel, C1 layer, alexnet
    for i_cond = 1:ncond
        switch rows{i_row}
            case 'pixel' % Leibo model - raw pixel
                y = nan(nview,nid*npc);
                tmp = cellfun(@(M) M(:), train_imgs{1}, 'UniformOutput', false);
                train_mat = double(cell2mat(tmp'));
                tmp = cellfun(@(M) M(:), test_imgs{i_cond}, 'UniformOutput', false);
                test_mat = double(cell2mat(tmp'));
                corr_val = nan(nexemplar,1);
                avg_face = mean(train_mat,2);
                % running PCA
                X = train_mat - avg_face * ones(1,size(train_mat,2));
                for i_exemplar = 1:nexemplar
                    count = 0;
                    imgs = test_mat(:,i_exemplar : nexemplar : nexemplar *nview);
                    for i_id = 1:nid
                        tmp_id = X(:,(i_id - 1) * ntrview + 1:i_id * ntrview);
                        [U,S,V] = pca(tmp_id);
                        for i_pc = 1:npc  % all 39 PCs
                            count = count + 1;
                            w = S(:,i_pc);
                            for i_view = 1:nview % number of view
                                y(i_view,count) = power(dot(w(:),imgs(:,i_view)),2);
                            end
                        end
                    end
                    % measure RDM and msvt
                    RDM_ = squareform(pdist(y,'correlation'));
                    corr_val(i_exemplar,1) = msvt_index(RDM_);
                end
                
                
            case 'c1'  % Leibo model- C1 layer
                y = nan(nview,nexemplar);
                train_mat = train_imgs_C1_layer;
                test_mat = test_imgs_C1_layer{i_cond};
                corr_val = nan(nexemplar,1);
                avg_face = mean(train_mat,2);
                % running PCA
                X = train_mat - avg_face * ones(1,size(train_mat,2));
                for i_exemplar = 1:nexemplar
                    count = 0;
                    imgs = test_mat(:,i_exemplar : nexemplar : nexemplar *nview);
                    for i_id = 1:nid
                        tmp_id = X(:,(i_id - 1) * ntrview + 1:i_id * ntrview);
                        [U,S,V] = pca(tmp_id);
                        for i_pc = 1:npc  % all 39 PCs
                            count = count + 1;
                            w = S(:,i_pc);
                            for i_view = 1:nview % number of view
                                y(i_view,count) = power(dot(w(:),imgs(:,i_view)),2);
                            end
                        end
                    end
                    % measure RDM and msvt
                    RDM_ = squareform(pdist(y,'correlation'));
                    corr_val(i_exemplar,1) = msvt_index(RDM_);
                end
                
            case 'alexnet' % AlexNet model
                corr_val = nan(nexemplar,1);
                for i_layer = 10 % 1:length(layers) % fc6 layer
                    imgs_res_trained = [];
                    object_category = test_imgs{i_cond};
                    % exemplar
                    sample = [];
                    for i_exemplar = 1 : nexemplar
                        sample = object_category(i_exemplar : nexemplar: nview * nexemplar);
                        obj_res_trained = [];
                        for i_view = 1 : nview
                            img = sample{i_view};
                            if ~isa(img,'uint8');img = im2uint8(img);end
                            % trained network
                            imgRGB = cat(3,img,img,img);
                            img_ = single(imgRGB) ; % note: 0-255 range
                            img_ = imresize(img_, trained_net.Layers(1).InputSize(1:2)) ;
                            img_res_trained = activations(trained_net,img_,layers(i_layer),'ExecutionEnvironment','gpu');
                            obj_res_trained = [obj_res_trained squeeze(img_res_trained(:))];
                            
                        end
                        imgs_res_trained = [imgs_res_trained, obj_res_trained];
                    end
                    for i_obj = 1:nexemplar
                        % compute dissimilarity matrix
                        obj_res_trained_ = imgs_res_trained(:,1 + ((i_obj - 1) * nview) : i_obj * nview);
                        RDM_ = 1 - corr(obj_res_trained_,'rows','complete');
                        corr_val(i_obj,1) = msvt_index(RDM_);
                        disp(['---> layer: ' name_layers{i_layer}])
                    end
                end
        end
        
        % plot setting
        axes(h(i_row))
        
        xlim([0.75,2.25])
        xtick = [1,2];
        
        h_ = swarmchart(i_cond*ones(1,nexemplar),corr_val,30,'filled','MarkerFaceColor',...
            colors(i_cond,:),'MarkerEdgeColor',[1,1,1],'LineWidth',0.2,'MarkerFaceAlpha',0.6,...
            'MarkerEdgeAlpha',0.5);
        hold on
        scatter(i_cond,median(corr_val),50,'MarkerFaceColor',...
            [1,1,1],'MarkerEdgeColor',colors(i_cond,:))
        h_.XJitterWidth = 0.2;
        h_.XJitter = 'rand';

    end
        set(gca,'xtick',xtick,'xticklabel',xticklabel,'TickLabelInterpreter','none');
        ylim([-1,1])
        box off;
        yline(0,'Color',[0.8,0.8,0.8],'LineStyle','--')
        
        h(i_row).TickDir = 'out';
        h(i_row).TickLength = [0.005,0.005];
        h(i_row).Title.String = titles{i_row};
        h(i_row).Clipping = 'off';
end
axes(h(1))
ylabel({'Mirror symmetric';'viewpoint tuning'})

