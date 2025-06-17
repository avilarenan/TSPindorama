
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

python -u run.py \
	--is_training 1 \
	--root_path ./data/ETT/ \
	--data_path ETTh2.csv \
	--model_id wpmixer \
	--model $model_name \
	--task_name long_term_forecast \
	--data ETTh2 \
	--seq_len 512 \
	--pred_len 96 \
	--label_len 0 \
	--d_model 256 \
	--patch_len 16 \
	--batch_size 256 \
	--learning_rate 0.000466278 \
	--lradj type3 \
	--dropout 0.0 \
	--patience 12 \
	--train_epochs 30 \
	--use_amp

python -u run.py \
	--is_training 1 \
	--root_path ./data/ETT/ \
	--data_path ETTh2.csv \
	--model_id wpmixer \
	--model $model_name \
	--task_name long_term_forecast \
	--data ETTh2 \
	--seq_len 512 \
	--pred_len 192 \
	--label_len 0 \
	--d_model 256 \
	--patch_len 16 \
	--batch_size 256 \
	--learning_rate 0.000294929 \
	--lradj type3 \
	--dropout 0.0 \
	--patience 12 \
	--train_epochs 30 \
	--use_amp

python -u run.py \
	--is_training 1 \
	--root_path ./data/ETT/ \
	--data_path ETTh2.csv \
	--model_id wpmixer \
	--model $model_name \
	--task_name long_term_forecast \
	--data ETTh2 \
	--seq_len 512 \
	--pred_len 336 \
	--label_len 0 \
	--d_model 128 \
	--patch_len 16 \
	--batch_size 256 \
	--learning_rate 0.000617476 \
	--lradj type3 \
	--dropout 0.1 \
	--patience 12 \
	--train_epochs 30 \
	--use_amp

python -u run.py \
	--is_training 1 \
	--root_path ./data/ETT/ \
	--data_path ETTh2.csv \
	--model_id wpmixer \
	--model $model_name \
	--task_name long_term_forecast \
	--data ETTh2 \
	--seq_len 512 \
	--pred_len 720 \
	--label_len 0 \
	--d_model 128 \
	--patch_len 16 \
	--batch_size 256 \
	--learning_rate 0.000810205 \
	--lradj type3 \
	--dropout 0.4 \
	--patience 12 \
	--train_epochs 30 \
	--use_amp

