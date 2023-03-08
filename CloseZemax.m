function CloseZemax(TheApplication)

% theSystem.Close(false);
% TheApplication.CloseApplication();

TheApplication.PrimarySystem.Close(false);
TheApplication.CloseApplication();
LogMessage(strcat('Closed Zemax OpticStudio connection.'))
end