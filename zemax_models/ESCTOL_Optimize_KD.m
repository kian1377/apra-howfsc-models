function res = ESCTOL_Optimize_KD( args )

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
    % initialize ZOSAPI
    import ZOSAPI.*;
    TheSystem = TheApplication.PrimarySystem;   

    % directory for monte carlo cases
    path = 'C:\Users\KAJA\Documents\Kevin\03 STP\03 SCOT\02 Tolerancing\esc_optics_tolerancing\Kevin Tolerancing\STP_TOL_V11';
    targetPrefix = 'MC_T';
    files = dir([path,strcat('\',targetPrefix,'*.ZMX')]);
    num_files = size(files,1);     
    files = struct2cell(files);

    % directory for merit function
    path_MF = 'C:\Users\KAJA\Documents\Kevin\03 STP\03 SCOT\02 Tolerancing\esc_optics_tolerancing\Merit Function\STP_MR_V3.MF';
    
    % Define field coordinates
    Hx = [-1, 0, 1, -1, 0, 1, -1, 0, 1, 0.661, 0.5, 0.5, -0.661, -0.5, -0.5];
    Hy = [1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 1, -1, 0, 1, -1];
    
    tic;
    for j = 1:num_files
        % I like progress bars
        if j == 1
            wb1 = waitbar(0, sprintf('File 1 of %.0f\nFailed Cases: 0', num_files));
            wb2 = waitbar(0, sprintf('Loading File...'));
            pos_wb1=get(wb1, 'position');
            pos_wb2=[pos_wb1(1) pos_wb1(2)-(pos_wb1(4)*1.5) pos_wb1(3) pos_wb1(4)];
            set(wb2, 'position', pos_wb2, 'doublebuffer', 'on');
        else
            waitbar(j/ num_files, wb1, sprintf('File %.0f of %.0f\nFailed Cases: %.0f', j, num_files, num_fail));
            waitbar(0, wb2, sprintf('Loading File...'));
        end


        % get file index
        if numel(num2str(j)) == 1
            CallIndex = strcat('000',num2str(j));
        elseif numel(num2str(j)) == 2
            CallIndex = strcat('00',num2str(j));
        elseif numel(num2str(j)) == 3
            CallIndex = strcat('0',num2str(j));
        end

        % load zos file
        path_file = strcat(path, '\', files{1, j});    
        TheSystem.LoadFile(path_file, false);
                     
        % grab obs pointing, M1, and M2 surfaces 
        obs = TheSystem.LDE.GetSurfaceAt(3);
        M1 = TheSystem.LDE.GetSurfaceAt(5);
        M2 = TheSystem.LDE.GetSurfaceAt(7);

        % pre-point M2
        M2.SurfaceData.TiltAbout_X = -M1.SurfaceData.TiltAbout_X;
        M2.SurfaceData.TiltAbout_Y = -M1.SurfaceData.TiltAbout_Y;
       
        % set optimization variables: M2 bulk motion
        M2.ThicknessCell.MakeSolveVariable();
        M2.SurfaceData.Par1.MakeSolveVariable(); 
        M2.SurfaceData.Par2.MakeSolveVariable(); 
        
        % initialize merit function and optimization loop
        TheMFE = TheSystem.MFE;
        TheMFE.LoadMeritFunction(strcat(path_MF))        
        TheMFE.CalculateMeritFunction;    

        % set system parameters: full aperture at 650 nm
        TheMCE = TheSystem.MCE;
        TheMCE.SetCurrentConfiguration(1);  % set to config 1 (full aperture)
        wave = TheSystem.SystemData.Wavelengths;
        wave.GetWavelength(1).Wavelength = 0.65; % set wavelength to 650 nm

        % set optimization parameters
        LocalOpt = TheSystem.Tools.OpenLocalOptimization();
        LocalOpt.Algorithm = ZOSAPI.Tools.Optimization.OptimizationAlgorithm.DampedLeastSquares;
        LocalOpt.Cycles = ZOSAPI.Tools.Optimization.OptimizationCycles.Fixed_10_Cycles;
        LocalOpt.NumberOfCores = 32;
        loopInd = 0;
        num_fail = 0;
        flag_WFE = ones(1,2) * 100;
                
        % optimization loop
        if ~isempty(LocalOpt)     
            % run until we meet spec on instrument fields
            while (46 < flag_WFE(1) * 650) || (46 < flag_WFE(2) * 650)
                % release M2 pointing after 100 cycles
                if loopInd == 10
                    M2.SurfaceData.Par3.MakeSolveVariable(); 
                    M2.SurfaceData.Par4.MakeSolveVariable(); 
                    disp('Release M2 pointing!')
                % release observatory pointing after 150 cycles
                elseif loopInd == 15
                    obs.SurfaceData.Par3.MakeSolveVariable();
                    obs.SurfaceData.Par4.MakeSolveVariable();
                    disp('Release observatory pointing!')
                % give up after 250 cycles
                elseif loopInd == 25
                    num_fail = num_fail + 1;
                    disp('Give up!')
                    break;
                
                end
                
                % run optimization               
                LocalOpt.RunAndWaitForCompletion();
                
                % use sub-aperture to evaluate instrument field WFE
                TheMCE.SetCurrentConfiguration(2);
                flag_WFE(1) = TheSystem.MFE.GetOperandValue(ZOSAPI.Editors.MFE.MeritOperandType.RWRE, 32, 0, Hx(10), Hy(10), 0, 0, 0, 0);
                flag_WFE(2) = TheSystem.MFE.GetOperandValue(ZOSAPI.Editors.MFE.MeritOperandType.RWRE, 32, 0, Hx(13), Hy(13), 0, 0, 0, 0);
                TheMCE.SetCurrentConfiguration(1);

                % update progress bar
                waitbar(loopInd / 25, wb2, sprintf('Optimizing File...\nCurrent Instrument RMS WFE: %6.2f nm', mean(flag_WFE) * 650))
                
                % update loop index
                loopInd = loopInd + 1;
            end
            % close optimization
            LocalOpt.Close();        
        end
        % update progress bar
        waitbar(1, wb2, sprintf('Saving File...'))
        
        % save optimized system
        TheSystem.SaveAs(strcat(path,'\OPT_', targetPrefix, CallIndex, '.ZMX'));   
    end
    toc;
    close(wb1);
    close(wb2);
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


