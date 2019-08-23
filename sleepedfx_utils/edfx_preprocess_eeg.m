function [data_chan_new] = edfx_preprocess_eeg(data_chan, fs, bpf, fc1, fc2)
    % data: eeg data matrix
    % fs: sampling frequency
    % bpf: band-pass filter (boolean)
    % fc1: cut-off frequency 1
    % fc2: cut-off frequency 2
    data_chan_new = data_chan;
    
    % ensure the signal is calibrated to microvolts
    if(max(data_chan_new) <= 10)
        disp('Signal calibrated!');
        data_chan_new = data_chan_new * 1000;
    end
    
    %% Band Pass 
    Fp1 = fc1; %(3Hz)
    Fp2 = fc2; %(40hz)
    Fst1 = fc1-2.5; % 0.5Hz 
    Fst2 = fc2+2.5; % 42.5Hz 
    Ap = 0.05;
    Ast = 60;
    
    if(bpf)
       bpfilter  = design(fdesign.bandpass('Fst1,Fp1,Fp2,Fst2,Ast1,Ap,Ast2', Fst1,Fp1,Fp2,Fst2,Ast,Ap,Ast,fs));
       data_chan_new = filtfilt(bpfilter.Numerator, 1, data_chan_new);
    end
    
    % zero-mean
    data_chan_new = data_chan_new - mean(data_chan_new);
    
end