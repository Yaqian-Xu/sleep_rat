function psg2edf(eeg1n,eeg2n,moven,varargin)
% function psg2edf(eeg1n,eeg2n,moven,datfile)
%   Generate EDF file with EEG and movement (EMG/ACC) signals for automated
%   sleep scoring with the SPINDLE algorithm of Miladinovic et al 2019 PLoS
%   Comput Biol.
%
%   eeg1n = name of first EEG channel in .mat PSG file ('EEGo', 'EEGc', 'EEGf')
%   eeg2n = name of second EEG channel in .mat PSG file ('EEGo', 'EEGc', 'EEGf')
%   moven = name of movement data in .mat PSG file ('EMG' or 'ACC');
%   datfile = name of .mat file with the PSG data; choose via ui if not specified
%
%   DR 10/2024

% data
if nargin<4
    [datfile,datpath] = uigetfile('*.mat','Choose data file');
    load(fullfile(datpath,datfile));
else
    datfile = varargin{1};
    load(datfile);
end 
idot = strfind(datfile,'.');
fnm = datfile(1:idot(1)-1); 

% eeg
if exist(eeg1n,'var')
    eval(['teeg = ' eeg1n '(:,1);']);
    eval(['eeg1 = ' eeg1n '(:,2);']);
else
    error('invalid first parameter');
end
if exist(eeg2n,'var')
    eval(['eeg2 = ' eeg2n '(:,2);']);
else
    error('invalid second parameter');
end
fs = round(1/median(diff(teeg)));
% wbcut = 50; % wideband frequency cutoff for plot (Hz)
% t = (1:length(eeg1))/fs/3600; % h
% [b,a] = butter(2,wbcut/(fs/2));
% EEG = filtfilt(b,a,EEG);

% movement
if matches(moven,'EMG') & exist('EMG','var')
    move = EMG(:,2);
elseif matches(moven,'ACC') & exist('acc_x','var')
    move = vecnorm([acc_x(:,2), acc_y(:,2), acc_z(:,2)]')';
    tmove = acc_x(:,1);
    move = interp1(tmove,move,teeg,'linear','extrap'); % resample ACC to match EEG
else
    error('invalid third parameter');
end

% psg data matrix
data = [eeg1, eeg2, move];
Nsamp = floor(length(eeg1)/fs)*fs;
data(Nsamp+1:end,:) = []; % truncate to last full second

% EDF header
hdr = edfheader("EDF+");
hdr.NumDataRecords = Nsamp/fs;
hdr.DataRecordDuration = seconds(1);
hdr.NumSignals = 3;
hdr.SignalLabels = ["EEG1" "EEG2" "EMG"];
hdr.PhysicalDimensions = repelem("mV",3); % Nora RHD data in mV
hdr.PhysicalMin = [-6.3898 -6.3898 -6.3898];    % [-5.5 -5.5 -5.5]; % Intan +/-5mV AC input range
hdr.PhysicalMax = [6.5 6.5 6.5];
hdr.DigitalMin = [-32767 -32767 -32767]; % Intan 16-bit ADC
hdr.DigitalMax = [32767 32767 32767];

% EDF file
edfwrite([fnm '.edf'],hdr,data,'InputSampleType','physical');
