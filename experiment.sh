#!/bin/bash

# export CUDA_VISIBLE_DEVICES=2
# long term forecasting
directory="./dataset/ETT-small"

# pred_lens=(96 132 336 720)
# model_names=(iTransformer TimeXer TimesNet TiDE DLinear PatchTST)

pred_lens=(96)
model_names=(PatchTST)

run_model() {
    local model_name=$1
    local data_filename=$2
    local dataset_name=$3
    local pred_len=$4

    # if [[ ! $data_filename == *"duplicated"* ]]; then
    #   echo "skipping"
    #   return 0
    # fi

    echo "Starting $model_name run with $data_filename and $dataset_name, with pred_len $pred_len"

    case "$model_name" in
        "TiDE")
            python -u run.py \
              --task_name long_term_forecast \
              --is_training 1 \
              --root_path ./dataset/ETT-small/ \
              --data_path $data_filename \
              --model_id "ETTh1_${pred_len}_96${dataset_name}" \
              --model $model_name \
              --data ETTh1 \
              --features MS \
              --seq_len 96 \
              --label_len 48 \
              --pred_len $pred_len \
              --e_layers 2 \
              --d_layers 2 \
              --enc_in 7 \
              --dec_in 7 \
              --c_out 8 \
              --d_model 256 \
              --d_ff 256 \
              --dropout 0.3 \
              --batch_size 512 \
              --learning_rate 0.1 \
              --patience 5 \
              --train_epochs 10
            ;;
        "TimeXer")
            python -u run.py \
              --task_name long_term_forecast \
              --is_training 1 \
              --root_path ./dataset/ETT-small/ \
              --data_path $data_filename \
              --model_id "ETTh1_${pred_len}_96${dataset_name}" \
              --model $model_name \
              --data ETTh1 \
              --features MS \
              --seq_len 96 \
              --label_len 48 \
              --pred_len $pred_len \
              --e_layers 1 \
              --factor 3 \
              --enc_in 7 \
              --dec_in 7 \
              --c_out 7 \
              --d_model 256 \
              --batch_size 4 \
              --des 'exp' \
              --itr 1
            ;;
        "TimesNet")
            python -u run.py \
              --task_name long_term_forecast \
              --is_training 1 \
              --root_path ./dataset/ETT-small/ \
              --data_path $data_filename \
              --model_id "ETTh1_${pred_len}_96${dataset_name}" \
              --model $model_name \
              --data ETTh1 \
              --features MS \
              --seq_len 96 \
              --label_len 48 \
              --pred_len $pred_len \
              --e_layers 2 \
              --d_layers 1 \
              --factor 3 \
              --enc_in 7 \
              --dec_in 7 \
              --c_out 7 \
              --d_model 16 \
              --d_ff 32 \
              --des 'Exp' \
              --itr 1 \
              --top_k 5 
            ;;
        "iTransformer")
            python -u run.py \
              --task_name long_term_forecast \
              --is_training 1 \
              --root_path ./dataset/ETT-small/ \
              --data_path $data_filename \
              --model_id "ETTh1_${pred_len}_96${dataset_name}" \
              --model $model_name \
              --data ETTh1 \
              --features MS \
              --seq_len 96 \
              --label_len 48 \
              --pred_len $pred_len \
              --e_layers 2 \
              --d_layers 1 \
              --factor 3 \
              --enc_in 7 \
              --dec_in 7 \
              --c_out 7 \
              --des 'Exp' \
              --d_model 128 \
              --d_ff 128 \
              --itr 1
            ;;
        "DLinear")
            python -u run.py \
              --task_name long_term_forecast \
              --is_training 1 \
              --root_path ./dataset/ETT-small/ \
              --data_path $data_filename \
              --model_id "ETTh1_${pred_len}_96${dataset_name}" \
              --model $model_name \
              --data ETTh1 \
              --features MS \
              --seq_len 96 \
              --label_len 48 \
              --pred_len $pred_len \
              --e_layers 2 \
              --d_layers 1 \
              --factor 3 \
              --enc_in 7 \
              --dec_in 7 \
              --c_out 7 \
              --des 'Exp' \
              --itr 1
            ;;  
        "PatchTST")
            python -u run.py \
              --task_name long_term_forecast \
              --is_training 1 \
              --root_path ./dataset/ETT-small/ \
              --data_path $data_filename \
              --model_id "ETTh1_${pred_len}_96${dataset_name}" \
              --model $model_name \
              --data ETTh1 \
              --features MS \
              --seq_len 96 \
              --label_len 48 \
              --pred_len 96 \
              --e_layers 1 \
              --d_layers 1 \
              --factor 3 \
              --enc_in 7 \
              --dec_in 7 \
              --c_out 7 \
              --des 'Exp' \
              --n_heads 2 \
              --itr 1
            ;;
        *)
            echo "Invalid option: $model_name"
            ;;
    esac
}

find "$directory" -maxdepth 1 -name "*.csv" -print0 | while IFS= read -r -d $'\0' file; do
  for model_name in "${model_names[@]}"; do
    for pred_len in "${pred_lens[@]}"; do
      dataset_name=$(basename "$file" .csv)
      data_filename=$(basename "$file")

      run_model "$model_name" "$data_filename" "$dataset_name" "$pred_len"
    done
  done
done
