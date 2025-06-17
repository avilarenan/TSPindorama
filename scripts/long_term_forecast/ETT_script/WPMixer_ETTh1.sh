#!/bin/bash

# Set the GPU to use
# export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

python -u run.py \
	--is_training 1 \
	--root_path ./data/ETT/ \
	--data_path ETTh1.csv \
	--model_id wpmixer \
	--model $model_name \
	--task_name long_term_forecast \
	--data ETTh1 \
	--seq_len 512 \
	--pred_len 96 \
	--label_len 0 \
	--d_model 256 \
	--patch_len 16 \
	--batch_size 256 \
	--learning_rate 0.000242438 \
	--lradj type3 \
	--dropout 0.4 \
	--patience 12 \
	--train_epochs 30 \
	--use_amp

python -u run.py \
	--is_training 1 \
	--root_path ./data/ETT/ \
	--data_path ETTh1.csv \
	--model_id wpmixer \
	--model $model_name \
	--task_name long_term_forecast \
	--data ETTh1 \
	--seq_len 512 \
	--pred_len 192 \
	--label_len 0 \
	--d_model 256 \
	--patch_len 16 \
	--batch_size 256 \
	--learning_rate 0.000201437 \
	--lradj type3 \
	--dropout 0.05 \
	--patience 12 \
	--train_epochs 30 \
	--use_amp

python -u run.py \
	--is_training 1 \
	--root_path ./data/ETT/ \
	--data_path ETTh1.csv \
	--model_id wpmixer \
	--model $model_name \
	--task_name long_term_forecast \
	--data ETTh1 \
	--seq_len 512 \
	--pred_len 336 \
	--label_len 0 \
	--d_model 256 \
	--patch_len 16 \
	--batch_size 256 \
	--learning_rate 0.000132929 \
	--lradj type3 \
	--dropout 0.0 \
	--patience 12 \
	--train_epochs 30 \
	--use_amp

python -u run.py \
	--is_training 1 \
	--root_path ./data/ETT/ \
	--data_path ETTh1.csv \
	--model_id wpmixer \
	--model $model_name \
	--task_name long_term_forecast \
	--data ETTh1 \
	--seq_len 512 \
	--pred_len 720 \
	--label_len 0 \
	--d_model 128 \
	--patch_len 16 \
	--batch_size 256 \
	--learning_rate 0.000239762 \
	--lradj type3 \
	--dropout 0.2 \
	--patience 12 \
	--train_epochs 30 \
	--use_amp
