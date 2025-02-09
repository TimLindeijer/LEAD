export CUDA_VISIBLE_DEVICES=0,1,2,3

# S-1-V-ADFTD-Sup
python -u run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-1-V-ADFTD-Sup --model LEAD --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# S-2-V-ADFTD-Sup
python -u run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-2-V-ADFTD-Sup --model LEAD --data MultiDatasets \
--training_datasets ADFTD,BrainLat-19 \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# S-3-V-ADFTD-Sup
python -u run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-3-V-ADFTD-Sup --model LEAD --data MultiDatasets \
--training_datasets ADFTD,BrainLat-19,CNBPM \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# S-4-V-ADFTD-Sup
python -u run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-4-V-ADFTD-Sup --model LEAD --data MultiDatasets \
--training_datasets ADFTD,BrainLat-19,CNBPM,Cognision-ERP-19 \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# S-5-V-ADFTD-Sup
python -u run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-5-V-ADFTD-Sup --model LEAD --data MultiDatasets \
--training_datasets ADFTD,BrainLat-19,CNBPM,Cognision-ERP-19,Cognision-rsEEG-19 \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15





