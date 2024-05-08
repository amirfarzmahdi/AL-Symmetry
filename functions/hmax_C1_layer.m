function [imgs_C1_layer] = hmax_C1_layer(images)
% measuring C1res
READPATCHESFROMFILE = 0; %use patches that were already computed
                         %(e.g., from natural images)
patchSizes = [4 8 12 16]; %other sizes might be better, maybe not
                          %all sizes are required
numPatchSizes = length(patchSizes);

%below the c1 prototypes are extracted from the images/ read from file
if ~READPATCHESFROMFILE
  tic
  numPatchesPerSize = 250; %more will give better results, but will
                           %take more time to compute
  cPatches = extractRandC1Patches(images, numPatchSizes, ...
      numPatchesPerSize, patchSizes); %fix: extracting from positive only 
                                      
  totaltimespectextractingPatches = toc;
else
  fprintf('reading patches');
  cPatches = load('PatchesFromNaturalImages250per4sizes','cPatches');
  cPatches = cPatches.cPatches;
end

%----Settings for Testing --------%
rot = [90 -45 0 45];
c1ScaleSS = [1:2:18];
RF_siz    = [7:2:39];
c1SpaceSS = [8:2:22];
minFS     = 7;
maxFS     = 39;
div = [4:-.05:3.2];
Div       = div;
%--- END Settings for Testing --------%

fprintf(1,'Initializing gabor filters -- full set...');
%creates the gabor filters use to extract the S1 layer
[fSiz,filters,c1OL,~] = init_gabor(rot, RF_siz, Div);
fprintf(1,'done\n');

% C1res 
  [imgs_C1_layer] = extractC2forcell(filters,fSiz,c1SpaceSS,c1ScaleSS,c1OL,cPatches,images,numPatchSizes);
end