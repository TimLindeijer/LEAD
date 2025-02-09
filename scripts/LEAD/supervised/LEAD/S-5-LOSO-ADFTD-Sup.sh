export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training
python -u run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-5-LOSO-ADFTD-Sup --model LEAD --data MultiDatasets \
--training_datasets ADFTD,CNBPM,Cognision-rsEEG-19,Cognision-ERP-19,BrainLat-19 \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len_list 4  --up_dim_list 76 --cross_val loso --swa \
--des 'Exp' --itr 65 --learning_rate 0.0001 --train_epochs 30 --patience 15
