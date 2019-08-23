############################################
# pretrained model is set to empty in each command
# Examples are given for the first cross-validation fold of SleepEDF-SC. Similar sript can be derived for other folds
# Likewise for other target databases
############################################

# 2-chan EEGEOG training
CUDA_VISIBLE_DEVICES="1,-1" python3 train_seqsleepnet.py --eeg_train_data "../../tf_data/sleepedf_sc/eeg/train_list_n1.txt" --eeg_eval_data "../../tf_data/sleepedf_sc/eeg/eval_list_n1.txt" --eeg_test_data "../../tf_data/sleepedf_sc/eeg/test_list_n1.txt" --eog_train_data "../../tf_data/sleepedf_sc/eog/train_list_n1.txt" --eog_eval_data "../../tf_data/sleepedf_sc/eog/eval_list_n1.txt" --eog_test_data "../../tf_data/sleepedf_sc/eog/test_list_n1.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --pretrained_model "" --out_dir './train_scratch_2chan/sleepedf_sc/n1/' --dropout_keep_prob_rnn 0.75 --seq_len 20 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size1 64 --evaluate_every 10
CUDA_VISIBLE_DEVICES="1,-1" python3 test_seqsleepnet.py --eeg_train_data "../../tf_data/sleepedf_sc/eeg/train_list_n1.txt" --eeg_test_data "../../tf_data/sleepedf_sc/eeg/test_list_n1.txt" --eog_train_data "../../tf_data/sleepedf_sc/eog/train_list_n1.txt" --eog_test_data "../../tf_data/sleepedf_sc/eog/test_list_n1.txt" --emg_train_data "" --emg_test_data "" --out_dir './train_scratch_2chan/sleepedf_sc/n1/' --dropout_keep_prob_rnn 0.75 --seq_len 20 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size1 64 --evaluate_every 10


# 1-chan EEG training
CUDA_VISIBLE_DEVICES="1,-1" python3 train_seqsleepnet.py --eeg_train_data "../../tf_data/sleepedf_sc/eeg/train_list_n1.txt" --eeg_eval_data "../../tf_data/sleepedf_sc/eeg/eval_list_n1.txt" --eeg_test_data "../../tf_data/sleepedf_sc/eeg/test_list_n1.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --pretrained_model "" --out_dir './train_scratch_1chan_eeg/sleepedf_sc/n1/' --dropout_keep_prob_rnn 0.75 --seq_len 20 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size1 64 --evaluate_every 10
CUDA_VISIBLE_DEVICES="1,-1" python3 test_seqsleepnet.py --eeg_train_data "../../tf_data/sleepedf_sc/eeg/train_list_n1.txt" --eeg_test_data "../../tf_data/sleepedf_sc/eeg/test_list_n1.txt" --eog_train_data "" --eog_test_data "" --emg_train_data "" --emg_test_data "" --out_dir './train_scratch_1chan_eeg/sleepedf_sc/n1/' --dropout_keep_prob_rnn 0.75 --seq_len 20 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size1 64 --evaluate_every 10


# 1-chan EOG finetuning
CUDA_VISIBLE_DEVICES="1,-1" python3 train_seqsleepnet.py --eeg_train_data "../../tf_data/sleepedf_sc/eog/train_list_n1.txt" --eeg_eval_data "../../tf_data/sleepedf_sc/eog/eval_list_n1.txt" --eeg_test_data "../../tf_data/sleepedf_sc/eog/test_list_n1.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --pretrained_model "" --out_dir './train_scratch_1chan_eog/sleepedf_sc/n1/' --dropout_keep_prob_rnn 0.75 --seq_len 20 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size1 64 --evaluate_every 10
CUDA_VISIBLE_DEVICES="1,-1" python3 test_seqsleepnet.py --eeg_train_data "../../tf_data/sleepedf_sc/eog/train_list_n1.txt" --eeg_test_data "../../tf_data/sleepedf_sc/eog/test_list_n1.txt" --eog_train_data "" --eog_test_data "" --emg_train_data "" --emg_test_data "" --out_dir './train_scratch_1chan_eog/sleepedf_sc/n1/' --dropout_keep_prob_rnn 0.75 --seq_len 20 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size1 64 --evaluate_every 10

