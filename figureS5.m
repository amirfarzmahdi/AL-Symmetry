% figure S5: measuring symmetry of the weights
% last update: October 28 2022
% Amirhossein Farzmahdi

close all
clear
clc

% add path
addpath(genpath('functions'))

% fixed random seed for regenerating same result
seed = 42; rng(seed)

% load network
trained_net = alexnet;

% create random network
untrained_net = fun_Initializeweight(trained_net,1,1); # adopted from Baek et al., 2021

% setting
nnet = 2;
nflip = 2; % horizontal and vertical
layers = [2, 6, 10, 12, 14, 17];
fc6_input_filter_size = [6, 6, 256]; % size of input to the fc6 layer
mean_corr_vals = nan(length(layers),nnet);
std_corr_vals = nan(length(layers),nnet);
name_layer = [{'conv1'};{'conv2'};{'conv3'};{'conv4'};{'conv5'};{'fc6'}];
weights_sym_corr = cell(length(layers),nnet,nflip);
colors = [0 0.4470 0.7410;0.8500 0.3250 0.0980];
x_pos = [1:2:11;(1:2:11)+0.5];
savefig = 1;
savemfile = 1;
y_labels = [{[{'Correlation between weights and'};{' horizontally flipped counterpart'}]},...
    {[{'Correlation between weights and'};{' vertically flipped counterpart'}]}];
figure_names = [{'figureS5_h'};{'figureS5_v'}];
for i_flip = 1 : nflip
    figure('units','inch','position',[0,0,3,2.25]);
    ax = axes;
    hold on
    for i_net = 1:nnet
        for i_layer = 1:length(layers)
            if i_net == 1 % trained network
                w =  trained_net.Layers(layers(i_layer),1).Weights;
            else % untrained network
                w =  untrained_net.Layers(layers(i_layer),1).Weights;
            end

            % check the weight size
            if length(size(w)) == 4
                nfilter = size(w,4);
            elseif length(size(w)) == 5
                w = cat(4,squeeze(w(:,:,:,:,1)),squeeze(w(:,:,:,:,2)));
                nfilter = (size(w,4));
            else
                w_ = w;
                nfilter = (size(w,1));
                w = nan([fc6_input_filter_size, nfilter]);
                for k = 1:nfilter
                    w(:,:,:,k) = reshape(w_(k,:),fc6_input_filter_size);
                end
            end

            % measure the correlation between weight tensor and its
            % horizontalluy (vertically) flipped counterpart
            filter_corrs = nan(nfilter,1);
            kernel_size = size(w(:,:,:,1),1);

            for i_filter = 1:nfilter
                orig_w = squeeze(w(:,:,:,i_filter));
                if i_flip == 1
                    % horizontal flip
                    flipped_w = fliplr(orig_w);
                    orig_w_half = orig_w(:,fix(1:kernel_size/2),:);
                    flipped_w_half = flipped_w(:,fix(1:kernel_size/2),:);
                else
                    % vertical flip
                    flipped_w = flipud(orig_w);
                    orig_w_half = orig_w(fix(1:kernel_size/2),:,:);
                    flipped_w_half = flipped_w(fix(1:kernel_size/2),:,:);
                end
                filter_corrs(i_filter,1) = corr(orig_w_half(:),flipped_w_half(:),'rows','complete');
            end

            mean_corr_vals(i_layer,i_net) = mean(filter_corrs);
            std_corr_vals(i_layer,i_net) = std(filter_corrs);
            weights_sym_corr{i_layer,i_net,i_flip} = filter_corrs;
            h(i_layer,i_net) = swarmchart(x_pos(i_net,i_layer)*ones(1,nfilter),filter_corrs,3,'filled','MarkerFaceColor',...
                colors(i_net,:),'MarkerEdgeColor',[1,1,1],'LineWidth',0.1,'MarkerFaceAlpha',0.5,...
                'MarkerEdgeAlpha',0.5);
            h(i_layer,i_net).XJitterWidth = 0.5;
            h(i_layer,i_net).XJitter = 'rand';
            hold on
            scatter(x_pos(i_net,i_layer),mean_corr_vals(i_layer,i_net),10,'MarkerFaceColor',colors(i_net,:),'MarkerEdgeColor',...
                [1,1,1]);
        end
    end

    % plot settings
    xlim([0,12]);
    ylim([-1,1]);
    xtick = mean(x_pos);
    xticklabel = name_layer;
    xtickangle(ax,45);
    yticks([-1,-0.5,0,0.5,1]);
    ylabel(y_labels{i_flip})

    set(gca,'xtick',xtick,'xticklabel',xticklabel,'TickLabelInterpreter','none');
    ax.TickDir = 'out';
    ax.TickLength = [0.005 0.005];
    ax.FontSize = 6;
    ax.FontName = 'Arial';
    h = legend([h(1,1),h(1,2)],{'trained','untrained'},'FontSize',4,'AutoUpdate','off','Box','off','FontName',...
        'Arial','Location','southeast');

    M = findobj(h,'type','patch'); % Find objects of type 'patch'
    set(M,'MarkerSize', 1)

end
