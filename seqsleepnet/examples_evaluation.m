%% 
% Examples on how to evaluate the performance
%% 
clear all
close all
clc

addpath('../metrics');

%% Example 1
% path to tensorflow experiments with SleepEDF-SC and the network output saved in
% test_ret.mat
% finetuning 2chan EEG+EOG experiment is used as the example here
ret_path = './tensorflow/seqsleepnet/finetune_all_2chan/sleepedf_sc/';

[acc, f1, kappa, mean_sens, mean_sel] = compute_sleepedfsc_performance(ret_path);


%% Example 2
% path to tensorflow experiments with SleepEDF-ST and the network output saved in
% test_ret.mat
% finetuning 2chan EEG+EOG experiment is used as the example here
ret_path = './tensorflow/seqsleepnet/finetune_all_2chan/sleepedf_st/';

[acc, f1, kappa, mean_sens, mean_sel] = compute_sleepedfsc_performance(ret_path);