%%% A script that does a standard 3D reconstruction of experimental data with ASTRA

clear all
clc
close all


addpath(genpath('~/astra-toolbox/matlab'))

%% reconstruction parameters

% directory where data is stored
dataDir = './JimsBreakfast/TestScan_06-02-20_fakecoffee_highres/'  % path to data


% specify source-detector distance (SDD) and source-object distance (SOD)
% (see "scan settings.txt")
SDD         = 528.014648
SOD         = 223.550781

%%% parameters of the preprocessing
binning        = 1;         % detector pixel binning
excludeLastPro = true       % exclude last projection angle which is often the same as the first one

%%% parameters of the reconstruction volume
nXYBase      = 64;    % number of voxels in XY direction
scaleFac     = 8;      % factor by which we upscale the geometry (higher = larger res)
angSubSamp   = 1;      % sub-sampling factor in angular direction
limAng       = 360;    % angular range (< 360 leads to limited view artifacts

%% check files and set up pre-processing

% Read in dark and flat fields
darkFieldFiles  = dir([dataDir, 'di*.tif']);
disp("test")
disp(size(darkFieldFiles));
flatFieldFiles  = dir([dataDir, 'io*.tif']);

detectorSize    = size(imread([dataDir darkFieldFiles(1).name]));

projectionFiles = dir([dataDir, 'scan_*.tif']);
angles          = linspace(0,2*pi, length(projectionFiles));
if(excludeLastPro)
    projectionFiles = projectionFiles(1:end-1);
    angles          = angles(1:end-1);
end
projectionFiles = projectionFiles(1:angSubSamp:end);
angles          = angles(1:angSubSamp:end);
angInd          = angles <= limAng/180*pi;
angles          = angles(angInd);
projectionFiles = projectionFiles(angInd);
nPro            = length(angles);

%% set up scanning and volume geometry

voxelSize       = 1 / scaleFac; % mm

% generate reconstruction geometry
nX      = nXYBase * scaleFac;
nY      = nX;
n       = nX*nY;
nXY     = [nX, nY];
volGeo  = astra_create_vol_geom(nX, nY);


% set up fan beam projection geometry
SDD         = SDD * scaleFac;
SOD         = SOD * scaleFac;
decPixelSz  = binning*0.074800 * scaleFac;

projGeo = astra_create_proj_geom('fanflat',  decPixelSz, detectorSize(2), angles, SOD, SDD - SOD);


%% if we compute reconstructions, read in and preprocess data

% read in both dark and flat
darkField = zeros(detectorSize);
for iDF = 1:length(darkFieldFiles)
    darkField = darkField + double(imread([dataDir, darkFieldFiles(iDF).name]));
end
darkField = darkField / length(darkFieldFiles);

flatField = zeros(detectorSize);
for iFF = 1:length(flatFieldFiles)
    flatField = flatField + double(imread([dataDir, flatFieldFiles(iFF).name]));
end
flatField = flatField / length(flatFieldFiles);


% read in data
data = zeros([nPro, detectorSize(2)]);

for iPro = 1:nPro

    data_i = double(imread([dataDir, projectionFiles(iPro).name]));

    % dark and flat field correction
    data_i = (data_i - darkField)./ (flatField - darkField);
    data(iPro,:) = data_i;

end

% reset values smaller or equal to 0
data(data <= 0) = min(data(data > 0));
% values larger than 1 are clipped to 1
data = min(data, 1);
% log data
data = - log(data);

% visualize data
figure();
imagesc(data);

%% compute FBP reconstruction

% Create astra objects for the reconstruction
recId  = astra_mex_data2d('create', '-vol', volGeo, 0);
sinoId = astra_mex_data2d('create', '-sino', projGeo, data);

% Set up the parameters for a reconstruction algorithm
% using the GPU
cfg = astra_struct('FBP_CUDA');

cfg.ProjectionDataId     = sinoId;
cfg.ReconstructionDataId = recId;

% Create the algorithm object from the configuration structure
algId                    = astra_mex_algorithm('create', cfg);

% Run FBP
astra_mex_algorithm('run', algId);

% Get the result
FBP_rec  = astra_mex_data2d('get', recId);

% clipp negative values
FBP_rec(FBP_rec < 0) = 0;

% Clean up. Note that GPU memory is tied up in the algorithm object,
% and main RAM in the data objects.
astra_mex_algorithm('delete', algId);
astra_mex_data2d('delete', sinoId, recId);


%% compute SIRT reconstruction

% Create astra objects for the reconstruction
recId  = astra_mex_data2d('create', '-vol', volGeo, 0);
sinoId = astra_mex_data2d('create', '-sino', projGeo, data);

% Set up the parameters for a reconstruction algorithm
% using the GPU
cfg = astra_struct('SIRT_CUDA');
cfg.ProjectionDataId     = sinoId;
cfg.ReconstructionDataId = recId;
cfg.option.MinConstraint = 0;

% Create the algorithm object from the configuration structure
algId                    = astra_mex_algorithm('create', cfg);

% Run SIRT for 1000 iterations
astra_mex_algorithm('iterate', algId, 1000);

% Get the result
SIRT_rec  = astra_mex_data2d('get', recId);

% Clean up. Note that GPU memory is tied up in the algorithm object,
% and main RAM in the data objects.
astra_mex_algorithm('delete', algId);
astra_mex_data2d('delete', sinoId, recId);

%% visulization

figure();
subplot(1,2,1); imagesc(FBP_rec); axis square
subplot(1,2,2); imagesc(SIRT_rec); axis square

%% delete all data objects

%clear all
astra_mex_data2d('info');
