export CUDA_VISIBLE_DEVICES=0,1,2,3

# ADFTD Dataset
python -u run.py --method Medformer --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-ADFTD --model LEAD --data SingleDataset \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--patch_len_list 2,4,8 --no_channel_block \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# CNBPM Dataset
python -u run.py --method Medformer --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-CNBPM --model LEAD --data SingleDataset \
--training_datasets CNBPM \
--testing_datasets CNBPM \
--patch_len_list 2,4,8 --no_channel_block \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# Cognision-rsEEG Dataset
python -u run.py --method Medformer --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-Cognision-rsEEG --model LEAD --data SingleDataset \
--training_datasets Cognision-rsEEG \
--testing_datasets Cognision-rsEEG \
--patch_len_list 2,4,8 --no_channel_block \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# Cognision-ERP Dataset
python -u run.py --method Medformer --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-Cognision-ERP --model LEAD --data SingleDataset \
--training_datasets Cognision-ERP \
--testing_datasets Cognision-ERP \
--patch_len_list 2,4,8 --no_channel_block \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# BrainLat Dataset
python -u run.py --method Medformer --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-BrainLat --model LEAD --data SingleDataset \
--training_datasets BrainLat \
--testing_datasets BrainLat \
--patch_len_list 2,4,8 --no_channel_block \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15