% zscore function
% October 24 2022
% Amirhossein Farzmahdi

function [output] = zscore(input,dim)
% input: matrix feature x images
% ouput: zscored input
std_ = std(input,[],dim);
mean_ = mean(input,dim);
output = (input - mean_) ./ std_;
