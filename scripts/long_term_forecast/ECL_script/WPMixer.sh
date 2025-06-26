#!/bin/bash

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

python -u run.py \
	--model $model_name \
	--task_name long_term_forecast \
	--data ECL \
	--seq_len 512 \
	--pred_len 96 \
	--d_model 32 \
	--tfactor 3 \
	--dfactor 5 \
	--wavelet sym3 \
	--level 2 \
	--patch_len 16 \
	--stride 8 \
	--batch_size 32 \
	--learning_rate 0.00328086 \
	--lradj type3 \
	--dropout 0.1 \
	--embedding_dropout 0.0 \
	--patience 12 \
	--train_epochs 100

python -u run.py \
	--model $model_name \
	--task_name long_term_forecast \
	--data ECL \
	--seq_len 512 \
	--pred_len 192 \
	--d_model 32 \
	--tfactor 7 \
	--dfactor 5 \
	--wavelet coif5 \
	--level 3 \
	--patch_len 16 \
	--stride 8 \
	--batch_size 32 \
	--learning_rate 0.000493286 \
	--lradj type3 \
	--dropout 0.1 \
	--embedding_dropout 0.05 \
	--patience 12 \
	--train_epochs 100

python -u run.py \
	--model $model_name \
	--task_name long_term_forecast \
	--data ECL \
	--seq_len 512 \
	--pred_len 336 \
	--d_model 32 \
	--tfactor 5 \
	--dfactor 7 \
	--wavelet sym4 \
	--level 1 \
	--patch_len 16 \
	--stride 8 \
	--batch_size 32 \
	--learning_rate 0.002505375 \
	--lradj type3 \
	--dropout 0.2 \
	--embedding_dropout 0.05 \
	--patience 12 \
	--train_epochs 100

python -u run.py \
	--model $model_name \
	--task_name long_term_forecast \
	--data ECL \
	--seq_len 512 \
	--pred_len 720 \
	--d_model 32 \
	--tfactor 7 \
	--dfactor 8 \
	--wavelet db2 \
	--level 2 \
	--patch_len 16 \
	--stride 8 \
	--batch_size 32 \
	--learning_rate 0.001977516 \
	--lradj type3 \
	--dropout 0.1 \
	--embedding_dropout 0.0 \
	--patience 12 \
	--train_epochs 100