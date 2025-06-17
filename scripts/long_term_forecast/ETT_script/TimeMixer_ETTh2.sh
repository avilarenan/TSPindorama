export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 16 \
  --d_ff 32 \
  --learning_rate 0.01 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 16 \
  --d_ff 32 \
  --learning_rate 0.01 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 16 \
  --d_ff 32 \
  --learning_rate 0.01 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 16 \
  --d_ff 32 \
  --learning_rate 0.01 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2
