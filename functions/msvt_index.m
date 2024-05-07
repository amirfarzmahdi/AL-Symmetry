% mirror-symmetric viewpoint tuning index function
% October 24 2022
% Amirhossein Farzmahdi

function [output] = msvt_index(input)
    % input: RDM, views x views
    % ouput: mirror-symmetric viewpoint tuning index

    input_hflipped = fliplr(input);

    % index of central column
    indx = ceil(length(input)/2);

    % remove central column
    input(:,indx) = [];
    input_hflipped(:,indx) = [];

    % mirror symmetric viewpoint tuning index
    output = corr(input(:),input_hflipped(:),'type','Pearson');
