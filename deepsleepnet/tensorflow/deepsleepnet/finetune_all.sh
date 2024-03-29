############################################
# finetune_mode is set to 0 and a pretrained model is provided in each command
# Examples are given for the first cross-validation fold of SleepEDF-SC. Similar sript can be derived for other folds
# Likewise for other target databases
############################################

# 2-chan EEGEOG->EEGEOG finetuning
CUDA_VISIBLE_DEVICES="1,-1" python3 train_deepsleepnet.py --eeg_train_data "../../tf_data/sleepedf_sc/eeg/train_list_n1.txt" --eeg_eval_data "../../tf_data/sleepedf_sc/eeg/eval_list_n1.txt" --eeg_test_data "../../tf_data/sleepedf_sc/eeg/test_list_n1.txt" --eog_train_data "../../tf_data/sleepedf_sc/eog/train_list_n1.txt" --eog_eval_data "../../tf_data/sleepedf_sc/eog/eval_list_n1.txt" --eog_test_data "../../tf_data/sleepedf_sc/eog/test_list_n1.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --pretrained_model "../__pretrained_models/2chan_eegeog/checkpoint/pretrained_model.ckpt" --finetune_mode 0 --out_dir './finetune_all_2chan/sleepedf_sc/n1/' --dropout 0.5 --seq_len 20 --nhidden 512  --evaluate_every 10
CUDA_VISIBLE_DEVICES="1,-1" python3 test_deepsleepnet.py --eeg_train_data "../../tf_data/sleepedf_sc/eeg/train_list_n1.txt" --eeg_test_data "../../tf_data/sleepedf_sc/eeg/test_list_n1.txt" --eog_train_data "../../tf_data/sleepedf_sc/eog/train_list_n1.txt" --eog_test_data "../../tf_data/sleepedf_sc/eog/test_list_n1.txt" --emg_train_data "" --emg_test_data "" --out_dir './finetune_all_2chan/sleepedf_sc/n1/' --dropout 0.5 --seq_len 20 --nhidden 512  --evaluate_every 10


# 1-chan EEG->EEG finetuning
CUDA_VISIBLE_DEVICES="1,-1" python3 train_deepsleepnet.py --eeg_train_data "../../tf_data/sleepedf_sc/eeg/train_list_n1.txt" --eeg_eval_data "../../tf_data/sleepedf_sc/eeg/eval_list_n1.txt" --eeg_test_data "../../tf_data/sleepedf_sc/eeg/test_list_n1.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --pretrained_model "../__pretrained_models/1chan_eeg/checkpoint/pretrained_model.ckpt" --finetune_mode 0 --out_dir './finetune_all_1chan_eeg2eeg/sleepedf_sc/n1/' --dropout 0.5 --seq_len 20 --nhidden 512  --evaluate_every 10
CUDA_VISIBLE_DEVICES="1,-1" python3 test_deepsleepnet.py --eeg_train_data "../../tf_data/sleepedf_sc/eeg/train_list_n1.txt" --eeg_test_data "../../tf_data/sleepedf_sc/eeg/test_list_n1.txt" --eog_train_data "" --eog_test_data "" --emg_train_data "" --emg_test_data "" --out_dir './finetune_all_1chan_eeg2eeg/sleepedf_sc/n1/' --dropout 0.5 --seq_len 20 --nhidden 512  --evaluate_every 10


# 1-chan EOG->EOG finetuning
CUDA_VISIBLE_DEVICES="1,-1" python3 train_deepsleepnet.py --eeg_train_data "../../tf_data/sleepedf_sc/eog/train_list_n1.txt" --eeg_eval_data "../../tf_data/sleepedf_sc/eog/eval_list_n1.txt" --eeg_test_data "../../tf_data/sleepedf_sc/eog/test_list_n1.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --pretrained_model "../__pretrained_models/1chan_eog/checkpoint/pretrained_model.ckpt" --finetune_mode 0 --out_dir './finetune_all_1chan_eog2eog/sleepedf_sc/n1/' --dropout 0.5 --seq_len 20 --nhidden 512  --evaluate_every 10
CUDA_VISIBLE_DEVICES="1,-1" python3 test_deepsleepnet.py --eeg_train_data "../../tf_data/sleepedf_sc/eog/train_list_n1.txt" --eeg_test_data "../../tf_data/sleepedf_sc/eog/test_list_n1.txt" --eog_train_data "" --eog_test_data "" --emg_train_data "" --emg_test_data "" --out_dir './finetune_all_1chan_eog2eog/sleepedf_sc/n1/' --dropout 0.5 --seq_len 20 --nhidden 512  --evaluate_every 10


# 1-chan EEG->EOG finetuning
CUDA_VISIBLE_DEVICES="1,-1" python3 train_deepsleepnet.py --eeg_train_data "../../tf_data/sleepedf_sc/eog/train_list_n1.txt" --eeg_eval_data "../../tf_data/sleepedf_sc/eog/eval_list_n1.txt" --eeg_test_data "../../tf_data/sleepedf_sc/eog/test_list_n1.txt" --eog_train_data "" --eog_eval_data "" --eog_test_data "" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --pretrained_model "../__pretrained_models/1chan_eeg/checkpoint/pretrained_model.ckpt" --finetune_mode 0 --out_dir './finetune_all_1chan_eeg2eog/sleepedf_sc/n1/' --dropout 0.5 --seq_len 20 --nhidden 512  --evaluate_every 10
CUDA_VISIBLE_DEVICES="1,-1" python3 test_deepsleepnet.py --eeg_train_data "../../tf_data/sleepedf_sc/eog/train_list_n1.txt" --eeg_test_data "../../tf_data/sleepedf_sc/eog/test_list_n1.txt" --eog_train_data "" --eog_test_data "" --emg_train_data "" --emg_test_data "" --out_dir './finetune_all_1chan_eeg2eog/sleepedf_sc/n1/' --dropout 0.5 --seq_len 20 --nhidden 512  --evaluate_every 10