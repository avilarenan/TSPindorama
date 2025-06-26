#!/bin/bash

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

python -u run.py \
	--model $model_name \
	--task_name long_term_forecast \
	--data Traffic \
	--seq_len 1200 \
	--pred_len 96 \
	--d_model 16 \
	--tfactor 3 \
	--dfactor 5 \
	--wavelet db3 \
	--level 1 \
	--patch_len 16 \
	--stride 8 \
	--batch_size 16 \
	--learning_rate 0.0010385 \
	--lradj type3 \
	--dropout 0.05 \
	--embedding_dropout 0.05 \
	--patience 12 \
	--train_epochs 60

python -u run.py \
	--model $model_name \
	--task_name long_term_forecast \
	--data Traffic \
	--seq_len 1200 \
	--pred_len 192 \
	--d_model 32 \
	--tfactor 3 \
	--dfactor 5 \
	--wavelet db3 \
	--level 1 \
	--patch_len 16 \
	--stride 8 \
	--batch_size 16 \
	--learning_rate 0.000567053 \
	--lradj type3 \
	--dropout 0.05 \
	--embedding_dropout 0.0 \
	--patience 12 \
	--train_epochs 60

python -u run.py \
	--model $model_name \
	--task_name long_term_forecast \
	--data Traffic \
	--seq_len 1200 \
	--pred_len 336 \
	--d_model 32 \
	--tfactor 7 \
	--dfactor 7 \
	--wavelet bior3.1 \
	--level 1 \
	--patch_len 16 \
	--stride 8 \
	--batch_size 16 \
	--learning_rate 0.001026715 \
	--lradj type3 \
	--dropout 0.0 \
	--embedding_dropout 0.1 \
	--patience 12 \
	--train_epochs 50

python -u run.py \
	--model $model_name \
	--task_name long_term_forecast \
	--data Traffic \
	--seq_len 1200 \
	--pred_len 720 \
	--d_model 32 \
	--tfactor 7 \
	--dfactor 3 \
	--wavelet db3 \
	--level 1 \
	--patch_len 16 \
	--stride 8 \
	--batch_size 16 \
	--learning_rate 0.001496217 \
	--lradj type3 \
	--dropout 0.05 \
	--embedding_dropout 0.2 \
	--patience 12 \
	--train_epochs 60