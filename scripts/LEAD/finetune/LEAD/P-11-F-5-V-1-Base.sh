export CUDA_VISIBLE_DEVICES=0,1,2,3

# Finetune on all dataset but validate on one dataset

# ADFTD
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain_lead/LEAD/P-11-Base/ --task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-F-5-V-ADFTD-Base --model LEAD --data MultiDatasets \
--training_datasets ADFTD,CNBPM,Cognision-rsEEG-19,Cognision-ERP-19,BrainLat-19 \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15


# Brainlat
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain_lead/LEAD/P-11-Base/ --task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-F-5-V-BrainLat-Base --model LEAD --data MultiDatasets \
--training_datasets ADFTD,CNBPM,Cognision-rsEEG-19,Cognision-ERP-19,BrainLat-19 \
--testing_datasets BrainLat-19 \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15


# CNBPM
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain_lead/LEAD/P-11-Base/ --task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-F-5-V-CNBPM-Base --model LEAD --data MultiDatasets \
--training_datasets ADFTD,CNBPM,Cognision-rsEEG-19,Cognision-ERP-19,BrainLat-19 \
--testing_datasets CNBPM \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15


# Cognision-ERP
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain_lead/LEAD/P-11-Base/ --task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-F-5-V-Cognision-ERP-Base --model LEAD --data MultiDatasets \
--training_datasets ADFTD,CNBPM,Cognision-rsEEG-19,Cognision-ERP-19,BrainLat-19 \
--testing_datasets Cognision-ERP-19 \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15


# Cognision-rsEEG
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain_lead/LEAD/P-11-Base/ --task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-F-5-V-Cognision-rsEEG-Base --model LEAD --data MultiDatasets \
--training_datasets ADFTD,CNBPM,Cognision-rsEEG-19,Cognision-ERP-19,BrainLat-19 \
--testing_datasets Cognision-rsEEG-19 \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15


