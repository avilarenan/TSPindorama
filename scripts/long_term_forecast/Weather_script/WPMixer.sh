#!/bin/bash

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

python -u run.py \
	--model $model_name \
	--task_name long_term_forecast \
	--data Weather \
	--seq_len 512 \
	--pred_len 96 \
	--d_model 256 \
	--tfactor 3 \
	--dfactor 7 \
	--wavelet db3 \
	--level 2 \
	--patch_len 16 \
	--stride 8 \
	--batch_size 32 \
	--learning_rate 0.000913333 \
	--lradj type3 \
	--dropout 0.4 \
	--embedding_dropout 0.1 \
	--patience 12 \
	--train_epochs 60

python -u run.py \
	--model $model_name \
	--task_name long_term_forecast \
	--data Weather \
	--seq_len 512 \
	--pred_len 192 \
	--d_model 128 \
	--tfactor 3 \
	--dfactor 7 \
	--wavelet db3 \
	--level 1 \
	--patch_len 16 \
	--stride 8 \
	--batch_size 64 \
	--learning_rate 0.001379042 \
	--lradj type3 \
	--dropout 0.4 \
	--embedding_dropout 0.0 \
	--patience 12 \
	--train_epochs 60
		
python -u run.py \
	--model $model_name \
	--task_name long_term_forecast \
	--data Weather \
	--seq_len 512 \
	--pred_len 336 \
	--d_model 128 \
	--tfactor 7 \
	--dfactor 7 \
	--wavelet db3 \
	--level 2 \
	--patch_len 16 \
	--stride 8 \
	--batch_size 32 \
	--learning_rate 0.000607991 \
	--lradj type3 \
	--dropout 0.4 \
	--embedding_dropout 0.4 \
	--patience 12 \
	--train_epochs 60
	
python -u run.py \
	--model $model_name \
	--task_name long_term_forecast \
	--data Weather \
	--seq_len 512 \
	--pred_len 720 \
	--d_model 128 \
	--tfactor 7 \
	--dfactor 5 \
	--wavelet db2 \
	--level 1 \
	--patch_len 16 \
	--stride 8 \
	--batch_size 128 \
	--learning_rate 0.001470479 \
	--lradj type3 \
	--dropout 0.4 \
	--embedding_dropout 0.2 \
	--patience 12 \
	--train_epochs 60