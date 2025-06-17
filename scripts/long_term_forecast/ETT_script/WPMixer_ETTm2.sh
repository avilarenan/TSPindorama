
# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

python -u run.py \
	--is_training 1 \
	--root_path ./data/ETT/ \
	--data_path ETTm2.csv \
	--model_id wpmixer \
	--model $model_name \
	--task_name long_term_forecast \
	--data ETTm2 \
	--seq_len 512 \
	--pred_len 96 \
	--label_len 0 \
	--d_model 256 \
	--patch_len 48 \
	--batch_size 256 \
	--learning_rate 0.00076587 \
	--lradj type3 \
	--dropout 0.4 \
	--patience 12 \
	--train_epochs 80 \
	--use_amp

python -u run.py \
	--is_training 1 \
	--root_path ./data/ETT/ \
	--data_path ETTm2.csv \
	--model_id wpmixer \
	--model $model_name \
	--task_name long_term_forecast \
	--data ETTm2 \
	--seq_len 512 \
	--pred_len 192 \
	--label_len 0 \
	--d_model 256 \
	--patch_len 48 \
	--batch_size 256 \
	--learning_rate 0.000275775 \
	--lradj type3 \
	--dropout 0.2 \
	--patience 12 \
	--train_epochs 80 \
	--use_amp

python -u run.py \
	--is_training 1 \
	--root_path ./data/ETT/ \
	--data_path ETTm2.csv \
	--model_id wpmixer \
	--model $model_name \
	--task_name long_term_forecast \
	--data ETTm2 \
	--seq_len 512 \
	--pred_len 336 \
	--label_len 0 \
	--d_model 256 \
	--patch_len 48 \
	--batch_size 256 \
	--learning_rate 0.000234608 \
	--lradj type3 \
	--dropout 0.4 \
	--patience 12 \
	--train_epochs 80 \
	--use_amp
	
python -u run.py \
	--is_training 1 \
	--root_path ./data/ETT/ \
	--data_path ETTm2.csv \
	--model_id wpmixer \
	--model $model_name \
	--task_name long_term_forecast \
	--data ETTm2 \
	--seq_len 512 \
	--pred_len 720 \
	--label_len 0 \
	--d_model 256 \
	--patch_len 48 \
	--batch_size 256 \
	--learning_rate 0.001039536 \
	--lradj type3 \
	--dropout 0.4 \
	--patience 12 \
	--train_epochs 80 \
	--use_amp

