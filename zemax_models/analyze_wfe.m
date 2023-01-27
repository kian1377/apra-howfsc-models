clear all; close all; clc

theApplication = ConnectZemax();

% initialize ZOSAPI
import ZOSAPI.*;
theSystem = theApplication.PrimarySystem;   

% path to zos file
path = pwd;
file = 'jwst_segmented.zos';

% load zos file
zos_path = strcat(path, '\', file);    
theSystem.LoadFile(zos_path, false);

wave = theSystem.SystemData.Wavelengths.GetWavelength(1).Wavelength * 1e-6;

% get the MCE and set system configuration tio operate on
theMCE = theSystem.MCE;
theMCE.SetCurrentConfiguration(1);  % config 1 - full aperture

% get the Lens Data Editor so that properties can be altered
theLDE = theSystem.LDE;

% Alter the LDE as desired
theLDE.GetSurfaceAt(4).Thickness = -7169.1;

% Get Wavefront data
% open WFMap
WFMap = theSystem.Analyses.New_Analysis(ZOSAPI.Analysis.AnalysisIDM.WavefrontMap);

% adjust WFMap settings
WFMapSettings = WFMap.GetSettings();
WFMapSettings.Field.SetFieldNumber(1);
WFMapSettings.Sampling = ZOSAPI.Analysis.SampleSizes.S_512x512; % increase sampling

% apply settings 
WFMap.ApplyAndWaitForCompletion();

% get results
WFMapResults = WFMap.GetResults();
wfe_data = flipud(WFMapResults.DataGrids(1).Values.double);

% result is in waves so we convert to nm
wfe_data = wfe_data * wave;
imagesc(wfe_data)

theSystem.Close(false);
theApplication.CloseApplication();


