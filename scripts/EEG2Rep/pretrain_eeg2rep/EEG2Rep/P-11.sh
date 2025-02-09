export CUDA_VISIBLE_DEVICES=0,1,2,3

# Pretraining
python -u run.py --method EEG2Rep --task_name pretrain_eeg2rep --is_training 1 --root_path ./dataset/ --model_id P-11 --model EEG2Rep --data MultiDatasets \
--pretraining_datasets ADSZ,APAVA-19,ADFSU,AD-Auditory,TDBRAIN-19,TUEP,REEG-PD-19,PEARL-Neuro-19,Depression-19,REEG-SRM-19,REEG-BACA-19 \
--training_datasets ADFTD,CNBPM,Cognision-rsEEG-19,Cognision-ERP-19,BrainLat-19 \
--testing_datasets ADFTD,CNBPM,Cognision-rsEEG-19,Cognision-ERP-19,BrainLat-19 \
--e_layers 20 --batch_size 512 --n_heads 12 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0002 --train_epochs 50