function [acc, f1, kappa, mean_sens, mean_sel] = compute_sleepedfsc_performance(ret_path)

    seq_len = 20;
    Nfold = 20;
    yh = cell(Nfold,1);
    yt = cell(Nfold,1);
    mat_path = './mat/sleepedf_sc/';
    % load data split
    load('./data_split_sleepedf_sc.mat');

    for fold = 1 : Nfold
        fold
        test_s = test_sub{fold};
        sample_size = [];
        for i = 1 : numel(test_s)
            i
            for night = 1 : 2
                sname = ['n', num2str(test_s(i),'%02d'), '_', num2str(night), '_eeg.mat'];
                % subject 13 does not have 2 nights
                if(~exist([mat_path, sname], 'file'))
                    continue
                end
                load([mat_path,sname], 'label');
                % this is actual output of the network as we excluded those at the
                % recording ends which do not consitute a full sequence
                sample_size = [sample_size; numel(label) -  (seq_len - 1)]; 
                % pool ground-truth labels of all test subjects
                yt{fold} = [yt{fold}; double(label)];
            end
        end

        
        if(~exist([ret_path, 'n', num2str(fold),'/test_ret.mat'],'file'))
            disp('Returned file does not exist:')
            disp([ret_path, 'n', num2str(fold),'/test_ret.mat'])
        end
        
        load([ret_path, 'n', num2str(fold),'/test_ret.mat']);
        % as we shifted by one PSG epoch when generating sequences, L (sequence
        % length) decisions are available for each PSG epoch. This segment is
        % to aggregate the decisions to derive the final one.
        score_ = cell(1,seq_len);
        for n = 1 : seq_len
            score_{n} = softmax(squeeze(score(n,:,:)));
        end
        score = score_;
        clear score_;

        count = 0;
        for i = 1 : numel(test_s)
            for night = 1 : 2
                sname = ['n', num2str(test_s(i),'%02d'), '_', num2str(night), '_eeg.mat'];
                if(~exist([mat_path, sname], 'file'))
                    continue
                end
                count = count + 1;
                % start and end positions of current test subject's output
                start_pos = sum(sample_size(1:count-1)) + 1;
                end_pos = sum(sample_size(1:count-1)) + sample_size(count);
                score_i = cell(1,seq_len);
                for n = 1 : seq_len
                    score_i{n} = score{n}(start_pos:end_pos, :);
                    N = size(score_i{n},1);
                    % padding ones for those positions not constituting full
                    % sequences
                    score_i{n} = [ones(seq_len-1,5); score{n}(start_pos:end_pos, :)];
                    score_i{n} = circshift(score_i{n}, -(seq_len - n), 1);
                end

                % multiplicative probabilistic smoothing for aggregation
                % which equivalent to summation in log domain
                fused_score = log(score_i{1});
                for n = 2 : seq_len
                    fused_score = fused_score + log(score_i{n});
                end

                % the final output labels via likelihood maximization
                yhat = zeros(1,size(fused_score,1));
                for k = 1 : size(fused_score,1)
                    [~, yhat(k)] = max(fused_score(k,:));
                end

                % pool outputs of all test subjects
                yh{fold} = [yh{fold}; double(yhat')];
            end
        end
    end
    
    yh = cell2mat(yh);
    yt = cell2mat(yt);
    
    [acc, kappa, f1, ~, spec] = calculate_overall_metrics(yt, yh);
    [sens, sel]  = calculate_classwise_sens_sel(yt, yh);
    mean_sens = mean(sens);
    mean_sel = mean(sel);
end

