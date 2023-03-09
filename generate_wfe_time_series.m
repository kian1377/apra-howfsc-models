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

keys = ["decenter_x" "decenter_y" "decenter_z" "tilt_x" "tilt_y" "tilt_z"];

% misalign the primary according to the specified values
O = 4; % number of optics to be perturbed at each iteration;
for o=1:O
    m1_vals = [0 0 0 0 0 0];
    m1_vals(1:3) = m1_vals(1:3)*1e3; % put the decenter values into mm
    m1_misalignment = dictionary(keys, m1_vals);
    for i=1:N
        for j=1:N
            for k=1:N
                for l=1:N
                    for m=1:N
                        for n=1:N
theLDE.GetSurfaceAt(3).TiltDecenterData.BeforeSurfaceDecenterX = m1_misalignment("decenter_x");
theLDE.GetSurfaceAt(3).TiltDecenterData.BeforeSurfaceDecenterY = m1_misalignment("decenter_y");
theLDE.GetSurfaceAt(3).TiltDecenterData.BeforeSurfaceTiltX = m1_misalignment("tilt_x");
theLDE.GetSurfaceAt(3).TiltDecenterData.BeforeSurfaceTiltY = m1_misalignment("tilt_y");
theLDE.GetSurfaceAt(3).TiltDecenterData.BeforeSurfaceTiltZ = m1_misalignment("tilt_z");

theLDE.GetSurfaceAt(4).Thickness = m1_m2 + m1_misalignment("decenter_z");
theLDE.GetSurfaceAt(4).TiltDecenterData.BeforeSurfaceDecenterX = m2_misalignment("decenter_x");
theLDE.GetSurfaceAt(4).TiltDecenterData.BeforeSurfaceDecenterY = m2_misalignment("decenter_y");
theLDE.GetSurfaceAt(4).TiltDecenterData.BeforeSurfaceTiltX = m2_misalignment("tilt_x");
theLDE.GetSurfaceAt(4).TiltDecenterData.BeforeSurfaceTiltY = m2_misalignment("tilt_y");
theLDE.GetSurfaceAt(4).TiltDecenterData.BeforeSurfaceTiltZ = m2_misalignment("tilt_z");

theLDE.GetSurfaceAt(6).Thickness = m1_m2 + m1_misalignment("decenter_z");
theLDE.GetSurfaceAt(6).TiltDecenterData.BeforeSurfaceDecenterX = m3_misalignment("decenter_x");
theLDE.GetSurfaceAt(6).TiltDecenterData.BeforeSurfaceDecenterY = m3_misalignment("decenter_y");
theLDE.GetSurfaceAt(6).TiltDecenterData.BeforeSurfaceTiltX = m3_misalignment("tilt_x");
theLDE.GetSurfaceAt(6).TiltDecenterData.BeforeSurfaceTiltY = m3_misalignment("tilt_y");
theLDE.GetSurfaceAt(6).TiltDecenterData.BeforeSurfaceTiltZ = m3_misalignment("tilt_z");

theLDE.GetSurfaceAt(10).Thickness = m1_m2 + m1_misalignment("decenter_z");
theLDE.GetSurfaceAt(10).TiltDecenterData.BeforeSurfaceDecenterX = m1_misalignment("decenter_x");
theLDE.GetSurfaceAt(10).TiltDecenterData.BeforeSurfaceDecenterY = m1_misalignment("decenter_y");
theLDE.GetSurfaceAt(10).TiltDecenterData.BeforeSurfaceTiltX = m1_misalignment("tilt_x");
theLDE.GetSurfaceAt(10).TiltDecenterData.BeforeSurfaceTiltY = m1_misalignment("tilt_y");
theLDE.GetSurfaceAt(10).TiltDecenterData.BeforeSurfaceTiltZ = m1_misalignment("tilt_z");

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

% save the WFE data to a fits file
fname = 'wfe_on_axis.fits';
fitswrite(wfe_data, fname, 'WriteMode','overwrite')

fits.writeKey(fptr, 'PIXELSCL', 0.0102/1024, 'pixelscale of the pupil')
fits.writeKey(fptr, 'M1DECZ', m1_misalignment("decenter_z"))
fits.writeKey(fptr, 'M1DECX', m1_misalignment("decenter_x"))
fits.writeKey(fptr, 'M1DECY', m1_misalignment("decenter_y"))
fits.writeKey(fptr, 'M2DECZ', m2_misalignment("decenter_z"))
fits.writeKey(fptr, 'M2DECX', m2_misalignment("decenter_x"))
fits.writeKey(fptr, 'M2DECY', m2_misalignment("decenter_y"))
fits.writeKey(fptr, 'M3DECZ', m3_misalignment("decenter_z"))
fits.writeKey(fptr, 'M3DECX', m3_misalignment("decenter_x"))
fits.writeKey(fptr, 'M3DECY', m3_misalignment("decenter_y"))
fits.writeKey(fptr, 'M4DECZ', m4_misalignment("decenter_z"))
fits.writeKey(fptr, 'M4DECX', m4_misalignment("decenter_x"))
fits.writeKey(fptr, 'M4DECY', m4_misalignment("decenter_y"))

fits.writeKey(fptr, 'M1TILTX', m1_misalignment("tilt_x"))
fits.writeKey(fptr, 'M1TILTY', m1_misalignment("tilt_y"))
fits.writeKey(fptr, 'M1TILTZ', m1_misalignment("tilt_z"))
fits.writeKey(fptr, 'M2TILTX', m2_misalignment("tilt_x"))
fits.writeKey(fptr, 'M2TILTY', m2_misalignment("tilt_y"))
fits.writeKey(fptr, 'M2TILTZ', m2_misalignment("tilt_z"))
fits.writeKey(fptr, 'M3TILTX', m3_misalignment("tilt_x"))
fits.writeKey(fptr, 'M3TILTY', m3_misalignment("tilt_y"))
fits.writeKey(fptr, 'M31TILTZ', m3_misalignment("tilt_z"))
fits.writeKey(fptr, 'M4TILTX', m4_misalignment("tilt_x"))
fits.writeKey(fptr, 'M4TILTY', m4_misalignment("tilt_y"))
fits.writeKey(fptr, 'M4TILTZ', m4_misalignment("tilt_z"))
                        end
                    end
                end
            end
        end
    end
end
CloseZemax(theApplication)


