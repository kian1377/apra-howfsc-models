function ZPL2WFE( args )

    if ~exist('args', 'var')
        args = [];
    end
    
    % Initialize the OpticStudio connection
    TheApplication = InitConnection();
    if isempty(TheApplication)
        % failed to initialize a connection
        r = [];
    else
        try
            BeginApplication(TheApplication, args);
            CleanupConnection(TheApplication);
        catch err
            CleanupConnection(TheApplication);
            rethrow(err);
        end
    end
end


function BeginApplication(TheApplication, args)

% case number to evaluate
which_case = 6;

% initialize ZOSAPI
import ZOSAPI.*;
TheSystem = TheApplication.PrimarySystem;   

% path to zos file
path = pwd;
% file = 'STP_TMA_Mark-12F_DKim9_KD1_ClearAperture6p46m_Spider_M3Aperture_M2CA_M1ConicFixed_M2_M3MechDiaFixed.zos';
file = 'jwst_segmented.zos';

% load zos file
zos_path = strcat(path, '\', file);    
TheSystem.LoadFile(zos_path, false);

% directory for ZPL files
zpl_path = strcat(path, "\00_zpl\case ", num2str(which_case));
zpl_dir = dir(zpl_path);
zpl_dir(1:2) = [];

wb = waitbar(0, strcat("Evaluating case 1 of ", num2str(length(zpl_dir))));

% loop thru ZPL files
for i = 1:length(zpl_dir)

    % get name
    zpl_name = zpl_dir(i).name(1:end-4);

    % convert ZPL to fits
    [WFE_data, zern_data] = convert_ZPL(TheSystem, zpl_name);

    % save .fits results
    if strcmpi(zpl_name(1), 'F')
        fitswrite(WFE_data, strcat(path, "\01_fits\case ", num2str(which_case), "\FullModel_WFE_", zpl_name(end-2:end), ".fits"));
        fitswrite(zern_data, strcat(path, "\01_fits\case ", num2str(which_case), "\FullModel_ZERN_", zpl_name(end-2:end), ".fits"));
    elseif strcmpi(zpl_name(1), 'I')
        fitswrite(WFE_data, strcat(path, "\01_fits\case ", num2str(which_case), "\MechIso_WFE_", zpl_name(end-2:end), ".fits"));
        fitswrite(zern_data, strcat(path, "\01_fits\case ", num2str(which_case), "\MechIso_ZERN_", zpl_name(end-2:end), ".fits"));
    end
    
    % save .mat results
    if strcmpi(zpl_name(1), 'F')
        save(strcat(path, "\02_mat\case ", num2str(which_case), "\FullModel_", zpl_name(end-2:end), '.mat'), 'WFE_data', 'zern_data');
    elseif strcmpi(zpl_name(1), 'I')
        save(strcat(path, "\02_mat\case ", num2str(which_case), "\MechIso_", zpl_name(end-2:end), '.mat'), 'WFE_data', 'zern_data');
    end

    % reload zos file
    zos_path = strcat(path, '\', file);    
    TheSystem.LoadFile(zos_path, false);

    waitbar(i / length(zpl_dir), wb, strcat("Evaluating case ", num2str(i + 1)," of ", num2str(length(zpl_dir))));

end

close all

end

function [WFE_data, zern_data] = convert_ZPL(TheSystem, ZPL_name)

% get the MCE and set system parameters
TheMCE = TheSystem.MCE;
TheMCE.SetCurrentConfiguration(1);  % config 1 - full aperture
wave = TheSystem.SystemData.Wavelengths;
wave.GetWavelength(1).Wavelength = 1; % 1 um wavelength

% get MFE
TheMFE = TheSystem.MFE;

% create macro solver
MacroSolveDef = ZOSAPI.Editors.SolveType.ZPLMacro;

% get the LDE and add a dummy surface
TheLDE = TheSystem.LDE;
Dummy = TheLDE.AddSurface();

% add the macro solve to the dummy surface
MacroSolve = Dummy.RadiusCell.CreateSolveType(MacroSolveDef);

% select and run macro
MacroSolve.Macro = ZPL_name;
Dummy.RadiusCell.SetSolveData(MacroSolve);

% get rid of the dummy surface or the WFMap won't work
TheLDE.RemoveSurfaceAt(12);
   
% open WFMap
WFMap = TheSystem.Analyses.New_Analysis(ZOSAPI.Analysis.AnalysisIDM.WavefrontMap);

% adjust WFMap settings
WFMapSettings = WFMap.GetSettings();
WFMapSettings.Field.SetFieldNumber(7); % 4/7 correspond to spec/coro fields
WFMapSettings.Sampling = ZOSAPI.Analysis.SampleSizes.S_512x512; % increase sampling

% apply settings 
WFMap.ApplyAndWaitForCompletion();    

% get results
WFMapResults = WFMap.GetResults();
WFE_data = WFMapResults.DataGrids(1).Values.double;

% result is in waves so we convert to nm
WFE_data = WFE_data * 1000;

% add zernike operands to the MFE
zern_data = zeros(230, 1);

operand = TheMFE.GetOperandAt(1);
operand.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.ZERN);
operand.GetCellAt(2).IntegerValue = 1; % term
operand.GetCellAt(4).IntegerValue = 5; % sampling (5 is 512x512)
operand.GetCellAt(5).IntegerValue = 7; % field
operand.GetCellAt(6).IntegerValue = 1; % type (1 is noll/standard)

for i = 2:length(zern_data)
    operand = TheMFE.InsertNewOperandAt(i);
    operand.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.ZERN);
    operand.GetCellAt(2).IntegerValue = i; % term
    operand.GetCellAt(4).IntegerValue = 5; % sampling (5 is 512x512)
    operand.GetCellAt(5).IntegerValue = 7; % field
    operand.GetCellAt(6).IntegerValue = 1; % type (1 is noll/standard)
end

% evaluate MFE
TheMFE.CalculateMeritFunction();

% get zernike coefficients from evaluated operands
for i = 1:length(zern_data)
    operand = TheMFE.GetOperandAt(i);
    zern_data(i) = operand.GetCellAt(12).DoubleValue;
end

% result is in waves so we convert to nm
zern_data = zern_data * 1000;

end

function app = InitConnection()

import System.Reflection.*;

% Find the installed version of OpticStudio.
zemaxData = winqueryreg('HKEY_CURRENT_USER', 'Software\Zemax', 'ZemaxRoot');
NetHelper = strcat(zemaxData, '\ZOS-API\Libraries\ZOSAPI_NetHelper.dll');
% Note -- uncomment the following line to use a custom NetHelper path
% NetHelper = 'C:\Users\Documents\Zemax\ZOS-API\Libraries\ZOSAPI_NetHelper.dll';
NET.addAssembly(NetHelper);

success = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize();
% Note -- uncomment the following line to use a custom initialization path
% success = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize('C:\Program Files\OpticStudio\');
if success == 1
    LogMessage(strcat('Found OpticStudio at: ', char(ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory())));
else
    app = [];
    return;
end

% Now load the ZOS-API assemblies
NET.addAssembly(AssemblyName('ZOSAPI_Interfaces'));
NET.addAssembly(AssemblyName('ZOSAPI'));

% Create the initial connection class
TheConnection = ZOSAPI.ZOSAPI_Connection();

% Attempt to create a Standalone connection

% NOTE - if this fails with a message like 'Unable to load one or more of
% the requested types', it is usually caused by try to connect to a 32-bit
% version of OpticStudio from a 64-bit version of MATLAB (or vice-versa).
% This is an issue with how MATLAB interfaces with .NET, and the only
% current workaround is to use 32- or 64-bit versions of both applications.
app = TheConnection.CreateNewApplication();
if isempty(app)
   HandleError('An unknown connection error occurred!');
end
if ~app.IsValidLicenseForAPI
    HandleError('License check failed!');
    app = [];
end

end

function LogMessage(msg)
disp(msg);
end

function HandleError(error)
ME = MXException(error);
throw(ME);
end

function  CleanupConnection(TheApplication)
% Note - this will close down the connection.

% If you want to keep the application open, you should skip this step
% and store the instance somewhere instead.
TheApplication.CloseApplication();
end