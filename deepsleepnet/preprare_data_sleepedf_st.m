clear all
close all
clc


%addpath('./seqsleepnet/');
addpath(genpath('../sleepedfx_utils/'));
addpath(genpath('../edf_reader/'));

% path to the SleepEDF expanded database 
%%
% SleepEDF expanded database can be download from https://physionet.org/content/sleep-edfx/1.0.0/
% This experiment on the ST subset was conducted with 22 placebo recordings. 
% Make sure that you refer to https://physionet.org/content/sleep-edfx/1.0.0/ST-subjects.xls
% to obtain correct recordings
%%
%data_path = '~/database/sleepedfx/sleep-telemetry/';
data_path = '~/Work/Dataset/SleepEDFx/';

%%
% meta information, such as light-off and light-on times to extract the
% in-bed parts data from the whole day-night recordings
% the meta information is provided in 
% ./sleepedfx_meta/
% following: 
% S. A. Imtiaz and E. Rodriguez-Villegas, ?An open-source toolbox for
% standardized use of PhysioNet Sleep EDF Expanded Database,? in Proc.
% EMBC, 2015, pp. 6014?6017.
%%
meta_path = '../sleepedfx_meta/';

% where mat data will be saved 
mat_path = './mat/sleepedf_st/';
if(~exist(mat_path, 'dir'))
    mkdir(mat_path);
end

Nsub = 20; % numer of subjects
Ncat = 5; % number of sleep stages

% Sampling rate of hypnogram (default is 30s)
epoch_time = 30;
fs = 100; % sampling frequency

% parameter for short-time Fourier Transform
win_size  = 2;
overlap = 1;
nfft = 2^nextpow2(win_size*fs);

% list of all subjects
listing = dir([data_path, 'ST7*']);

for i = 1 : numel(listing)
    disp(listing(i).name)
    target_dir = [data_path, listing(i).name, '/'];
    
    %[~,filename,~] = fileparts(listing(i).name);
    %[sub_id, night] = edfx_dir2sub(filename);
    %sub_id = sub_id + 1; % index 0 to 1
    
    % load edf data to get Fpz-Cz, and EOGhorizontal channels
    edf_file = [target_dir, listing(i).name, '-PSG.edf'];
    [header, edf] = edfreadUntilDone(edf_file);
    channel_names = header.label;
    
    for c = 1 : numel(channel_names)
        channel_names{c} = strtrim(channel_names{c});
    end
    chan_ind_eeg = find(ismember(channel_names, 'EEGFpzCz'));
    if(isempty(chan_ind_eeg))
        disp('EEG channel not found!');
        pause;
    end
    if(header.frequency(chan_ind_eeg) ~= fs)
        disp('EEG sampling frequency mismatched!');
        pause;
    end
    
    chan_ind_eog = find(ismember(channel_names, 'EOGhorizontal'));
    if(isempty(chan_ind_eog))
        disp('EOG channel not found!');
        pause;
    end
    if(header.frequency(chan_ind_eog) ~= fs)
        disp('EOG sampling frequency mismatched!');
        pause;
    end
    
    chan_data_eeg = edf(chan_ind_eeg, :);
    chan_data_eog = edf(chan_ind_eog, :);
    clear edf header channel_names
    
    % ensure the signal is calibrated to microvolts
    if(max(chan_data_eeg) <= 10)
        disp('Signal calibrated!');
        chan_data_eeg = chan_data_eeg * 1000;
    end
    if(max(chan_data_eog) <= 10)
        disp('EOG Signal calibrated!');
        chan_data_eog = chan_data_eog * 1000;
    end
    % zero-mean
    chan_data_eeg = chan_data_eeg - mean(chan_data_eeg);
    % zero-mean
    chan_data_eog = chan_data_eog - mean(chan_data_eog);
    
    % load hypnogram
    hyp_file = fullfile([meta_path, listing(i).name, '/info/', listing(i).name, '.txt']);
    hypnogram = edfx_load_hypnogram( hyp_file );
    
    % process times to determine the in-bed duration
    [chan_data_eeg, chan_data_eog, hypnogram] = edfx_process_time_2chan([meta_path, listing(i).name, '/'], ... 
        chan_data_eeg, chan_data_eog, hypnogram, epoch_time, fs);
    
    % discrete labels
    label = edfx_hypnogram2label(hypnogram);
    exc_ind = (label == 0); % excluding Unknown and non-score
    disp([num2str(sum(exc_ind)), ' epochs excluded.'])
    label(exc_ind) = [];
    
    % one-hot encoding
    y = zeros(numel(label), Ncat);
    for n = 1 : numel(label)
        y(n, label(n)) = 1;
    end
    
    label = single(label);
    y = single(y);
    
    % raw EEG input (for deepsleepnet)
    X_eeg_raw = buffer(chan_data_eeg, epoch_time*fs);
    X_eeg_raw = X_eeg_raw';
    X_eeg_raw(exc_ind, :) = []; % excluded Unknow and non-score epochs
    X = single(X_eeg_raw);
    save([mat_path, 'n', num2str(i,'%02d'), '_eeg.mat'], 'X', 'label', 'y', '-v7.3');
    
    
    % raw EOG input (for deepsleepnet)
    X_eog_raw = buffer(chan_data_eog, epoch_time*fs);
    X_eog_raw = X_eog_raw';
    X_eog_raw(exc_ind, :) = []; % excluded Unknow and non-score epochs
    X = single(X_eog_raw);
    save([mat_path, 'n', num2str(i,'%02d'), '_eog.mat'], 'X', 'label', 'y', '-v7.3');
    

    clear X label y
    clear X_eeg X_eog X_eeg_raw X_eog_raw
end