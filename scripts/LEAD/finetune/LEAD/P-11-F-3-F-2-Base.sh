export CUDA_VISIBLE_DEVICES=0,1,2,3

# Finetune only on public datasets

# Two-step Finetune

# First step: only finetune on three private datasets
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain_lead/LEAD/P-11-Base/ --task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-F-3-Base --model LEAD --data MultiDatasets \
--training_datasets CNBPM,Cognision-rsEEG-19,Cognision-ERP-19 \
--testing_datasets CNBPM,Cognision-rsEEG-19,Cognision-ERP-19 \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# Second step: finetune on two public datasets based on the first step fine-tuned model
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/finetune/LEAD/P-11-F-3-Base/ --task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-F-3-F-2-Base --model LEAD --data MultiDatasets \
--training_datasets ADFTD,BrainLat-19 \
--testing_datasets ADFTD,BrainLat-19 \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15



# Testing
# ADFTD
python -u run.py --method LEAD --task_name finetune --is_training 0 --root_path ./dataset/ --model_id P-11-F-3-F-2-Base --model LEAD --data MultiDatasets \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# BrainLat-19
python -u run.py --method LEAD --task_name finetune --is_training 0 --root_path ./dataset/ --model_id P-11-F-3-F-2-Base --model LEAD --data MultiDatasets \
--testing_datasets BrainLat-19 \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15