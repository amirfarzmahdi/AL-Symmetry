% figure 2D & S2: measuring mirror-symmetric viewpoint tuning index in
% trained/untrained networks
% last update: December 23 2023
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
load(images_dir+"imgs");
load(images_dir+"bw_masks");

% images setting
figure_names = [{'figureS19A'}];
nexemplar = 25;
nview = 9;
ncategory = 9;
nplot = 3;
nobj = ncategory * nexemplar;
img_size = [227, 227]; % AlexNet
ps = [];

% colormap
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

% matches the mean luminance and contrast of a set of images
imgs_lumMatch = reshape(lumMatch(cm_imgs(:),cm_bw_masks(:)),[nobj, nview]);

% model setting
trained_net = alexnet; % trained network
untrained_net = fun_Initializeweight(trained_net,1,1); % untrained network
layers = [1, 3, 5, 7, 9, 11, 13, 15, 16, 18, 21];
name_layers = [{'image'};{'conv1'};{'pool1'};{'conv2'};{'pool2'};...
    {'conv3'};{'conv4'};{'conv5'};{'pool5'};{'fc6'};{'fc7'}];

for i_layer = 1:length(layers) % number of layers
    imgs_res_trained = [];
    imgs_res_untrained = [];
    
    % category
    for i_category = 1 : ncategory
        object_category = imgs_lumMatch(:,i_category);
        
        % exemplar
        sample = [];
        for i_exemplar = 1 : nexemplar
            
            sample = object_category(i_exemplar : nexemplar: nview * nexemplar);
            obj_res_trained = [];
            obj_res_untrained = [];
            
            for i_view = 1 : nview
                img = sample{i_view};
                
                if ~isa(img,'uint8');img = im2uint8(img);end
                
                % trained network
                imgRGB = cat(3,img,img,img);
                img_ = single(imgRGB) ; % note: 0-255 range
                img_ = imresize(img_, trained_net.Layers(1).InputSize(1:2)) ;
                img_res_trained = activations(trained_net,img_,layers(i_layer),'ExecutionEnvironment','gpu');
                obj_res_trained = [obj_res_trained squeeze(img_res_trained(:))];
                
                % untrained network
                img_res_untrained = activations(untrained_net,img_,layers(i_layer),'ExecutionEnvironment','gpu');
                obj_res_untrained = [obj_res_untrained squeeze(img_res_untrained(:))];
            end
            imgs_res_trained = [imgs_res_trained, obj_res_trained];
            imgs_res_untrained = [imgs_res_untrained, obj_res_untrained];
        end
    end
    
    % figure19A: without z-score activation maps
    imgs_res_trained_zscored = imgs_res_trained; % zscore(imgs_res_trained,2);
    imgs_res_untrained_zscored = imgs_res_untrained; % zscore(imgs_res_untrained,2);
    
    for i_obj = 1:nobj
        % compute dissimilarity matrix
        
        % trained
        obj_res_trained_zscored = imgs_res_trained_zscored(:,1 + ((i_obj - 1) * nview) : i_obj * nview);
        RDM_trained = 1 - corr(obj_res_trained_zscored,'rows','complete');
        
        msvt_index_values{1}(i_obj,i_layer) = msvt_index(RDM_trained);
        
        % untrained
        obj_res_untrained_zscored = imgs_res_untrained_zscored(:,1 + ((i_obj - 1) * nview) : i_obj * nview);
        RDM_untrained = 1 - corr(obj_res_untrained_zscored,'rows','complete');
        
        msvt_index_values{2}(i_obj,i_layer) = msvt_index(RDM_untrained);
        
        disp(['---> layer: ' name_layers{i_layer}...
            '   ' '---> ' num2str(names{i_obj})])
    end
end

% difference between trained and untrained msvt index values
msvt_index_values{3} = msvt_index_values{1} - msvt_index_values{2};

for i_plot = 1:1%nplot
    
    figure;
    ax = axes;
    hold on;
    % compute mean and std per category
    mean_catg_index = zeros(ncategory,length(layers));
    std_catg_index = zeros(ncategory,length(layers));
    for i_category = 1:ncategory
        for i_layer = 1:length(layers)
            index = ((i_category - 1) * nexemplar) + 1 : i_category * nexemplar;
            catg_corr = msvt_index_values{i_plot}(index,i_layer);
            mean_catg_index(i_category,i_layer) = mean(catg_corr);
            std_catg_index(i_category,i_layer) = std(catg_corr)/sqrt(nexemplar);
            
            % measuring p-values using ranksum test
            if i_layer ~= 1 && i_plot == 3
                [ps(i_category,i_layer-1), ~] = ranksum(msvt_index_values{1}(index,i_layer),...
                    msvt_index_values{2}(index,i_layer),'alpha',0.05);
            end
        end
        
        h_(i_category) = plot(1:length(layers),mean_catg_index(i_category,:),'Marker','o','MarkerEdgeColor',cmap(i_category,:),...
            'LineWidth',1,'MarkerSize',3,'MarkerFaceColor',cmap(i_category,:),'Color',cmap(i_category,:),'LineStyle', '-');
        errorbar(1:length(layers),mean_catg_index(i_category,:),std_catg_index(i_category,:),'Color',cmap(i_category,:),'CapSize',0,...
            'LineWidth',1,'HandleVisibility','off','LineStyle', '-');
    end
    
    % plot parameters
    xtick = 1:length(layers);
    xticklabel = name_layers;
    xtickangle(ax,45);
    set(gca,'xtick',xtick,'xticklabel',xticklabel,'TickLabelInterpreter','none');
    
    opt = [];
    opt.XLabel = '';
    opt.YLabel = 'Mirror-symmetric viewpoint tuning';
    opt.XTick = 1:length(layers);
    opt.XLim = [0.9 length(layers)];
    opt.Colors = cmap;
    opt.ShowBox = 'off';
    opt.LineWidth = 1 * ones(1,ncategory);
    opt.AxisLineWidth = 0.8;
    opt.Markers = repelem({'o'},1,ncategory);
    opt.Legend = {'car','boat','face','chair','airplane','tool','animal','fruit','flower'}; % legends
    opt.XMinorTick = 'off';
    opt.YMinorTick = 'off';
    opt.TickDir = 'out';
    opt.TickLength = [0.005 0.005];
    opt.FontSize = 6;
    opt.FontName = 'Arial';
    opt.LegendLoc = 'southeast';
    opt.BoxDim = [3 2.25];% maximum: 8.75 7.5
    
    if i_plot == 1 || i_plot == 2
        opt.YTick = [-0.5, 0, 0.5, 1];
        opt.YLim = [-0.75, 1];
        opt.LineStyle = {'-', '-', '-','-','-','-','-','-','-'};
        % create the plot
        setPlotProp(opt);
    else
        opt.YTick = [-0.5, 0, 0.5, 1, 1.5];
        opt.YLim = [-0.2, 1.5];
        opt.LineStyle = {'--', '--', '--','--','--','--','--','--','--'};
        % create the plot
        setPlotProp(opt);
        
        % correction for multiple comparisons
        [p_bf_fdr, ~, ~] = fdr_bh(ps,0.05, 'dep', 'yes');
        
        % add zero column for the input layer
        p_bf_fdr = [zeros(ncategory,1), p_bf_fdr];
        
        % significant category and layer
        p_bf_fdr(p_bf_fdr == 0) = nan;
        h = (p_bf_fdr .* mean_catg_index)';
        
        % showing significant layers
        plot(h,'Marker','o','LineWidth',1,'MarkerSize',3,'LineStyle',...
                'none','MarkerEdgeColor',[0.5,0.5,0.5]);
    end
    
    % set legend
%     legend(h_,'Location','best');
    legend('hide');
    
end