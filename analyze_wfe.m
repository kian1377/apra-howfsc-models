clear all; close all; clc
import matlab.io.*

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

keys = ["decenter_x" "decenter_y" "decenter_z" "tilt_x" "tilt_y" "tilt_z"];
m1_vals = [0 0 0 0 0 0];
m1_vals(1:3) = m1_vals(1:3)*1e3;
m1_misalignment = dictionary(keys, m1_vals);

% misalign the primary according to the specified values
theLDE.GetSurfaceAt(3).Thickness = m1_m2 + m1_misalignment("decenter_z");
theLDE.GetSurfaceAt(3).TiltDecenterData.BeforeSurfaceDecenterX = m1_misalignment("decenter_x");
theLDE.GetSurfaceAt(3).TiltDecenterData.BeforeSurfaceDecenterY = m1_misalignment("decenter_y");
theLDE.GetSurfaceAt(3).TiltDecenterData.BeforeSurfaceTiltX = m1_misalignment("tilt_x");
theLDE.GetSurfaceAt(3).TiltDecenterData.BeforeSurfaceTiltY = m1_misalignment("tilt_y");
theLDE.GetSurfaceAt(3).TiltDecenterData.BeforeSurfaceTiltZ = m1_misalignment("tilt_z");

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
wfe_data(isnan(wfe_data)) = 0;

% result is in waves so we convert to nm
wfe_data = wfe_data * wave;
imagesc(wfe_data)
colorbar('eastoutside', 'FontSize',12)
axis image
% imshow(wfe_data, [])

CloseZemax(theApplication)

% save the WFE data to a fits file
% fname = 'C:\Users\Kian\Documents\data-files\time-series-wfe\wfe_on_axis.fits';
fname = '!wfe_on_axis.fits';
fitswrite(wfe_data, fname, 'WriteMode','overwrite')
% fptr = fits.createFile(fname);
% fits.writeKey(fptr, 'PIXELSCL', 0.0102/1024, 'pixelscale of the pupil')
% fits.closeFile(fptr)


