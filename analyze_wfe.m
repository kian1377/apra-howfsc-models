clear all; close all; clc

theApplication = ConnectZemax();

% initialize ZOSAPI
import ZOSAPI.*;
theSystem = theApplication.PrimarySystem;   

% path to zos file
% path = pwd;
% file = 'luvoir_B_reduced.zos';

path = 'C:\Users\Kian\Desktop\zemax_models';
file = 'OFFAXIS_TMA.zos';

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
m1_m2 = theLDE.GetSurfaceAt(3).Thickness;
m2_m3 = theLDE.GetSurfaceAt(4).Thickness + theLDE.GetSurfaceAt(5).Thickness;
m3_fsm = theLDE.GetSurfaceAt(6).Thickness;
fsm_fp = theLDE.GetSurfaceAt(10).Thickness;

misalignment = 
theLDE.GetSurfaceAt(3).Thickness = m1_m2 + 1e-3;

% sysField = theSystem.SystemData.Fields;

% Get Wavefront data
% open WFMap
WFMap = theSystem.Analyses.New_Analysis(ZOSAPI.Analysis.AnalysisIDM.WavefrontMap);

% adjust WFMap settings
WFMapSettings = WFMap.GetSettings();
WFMapSettings.Field.SetFieldNumber(1);
WFMapSettings.Sampling = ZOSAPI.Analysis.SampleSizes.S_1024x1024; % increase sampling

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

% save the WFE data to a fits file
wfe_data(isnan(wfe_data)) = 0;
fname = 'wfe_on_axis.fits';
fitswrite(wfe_data, fname, 'WriteMode','overwrite')




