#!/bin/bash

# export CUDA_VISIBLE_DEVICES=2
# long term forecasting
directory="./dataset/ETT-small"

# pred_lens=(96 192 336 720)
# model_names=(iTransformer TimeXer TimesNet PatchTST)
# Dlinear and TiDE are channel independent, so exogenous features won't change the accuracy of predicting the target when in mode "MS"

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
            # From TSLib https://github.com/thuml/Time-Series-Library
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
            # From TSLib https://github.com/thuml/Time-Series-Library

            declare -A d_model_tune
            d_model_tune[96] = 512
            d_model_tune[192] = 128 # NOTE: does this make sense? only one different
            d_model_tune[336] = 512
            d_model_tune[720] = 512

            declare -A d_ff_tune
            d_ff_tune[96] = 512
            d_ff_tune[192] = 128 # NOTE: does this make sense? inflection here
            d_ff_tune[336] = 512
            d_ff_tune[720] = 2048 # inferred by default value

            declare -A batch_size_tune
            batch_size_tune[96] = 32 # inferred by default value
            batch_size_tune[192] = 4 # NOTE: does this make sense? inflection here
            batch_size_tune[336] = 32
            batch_size_tune[720] = 128

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
              --factor 3 \
              --enc_in 7 \
              --dec_in 7 \
              --c_out 7 \
              --d_model ${d_model_tune[$pred_len]} \
              --d_ff ${d_ff_tune[$pred_len]} \
              --batch_size ${batch_size_tune[$batch_size_tune]} \
              --des 'Exp' \
              --itr 1
              
            ;;
        "TimesNet")
            # From TSLib https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
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
        "DLinear")
            # From TSLib https://github.com/thuml/Time-Series-Library
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
            # From TSLib https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh
            declare -A n_heads_tune
            n_heads_tune[96] = 2
            n_heads_tune[192] = 8
            n_heads_tune[336] = 8
            n_heads_tune[720] = 16
            
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
              --d_layers 1 \
              --factor 3 \
              --enc_in 7 \
              --dec_in 7 \
              --c_out 7 \
              --des 'Exp' \
              --n_heads ${n_heads_tune[$pred_len]} \
              --itr 1
            ;;
        "iTransformer")
            # From https://github.com/thuml/iTransformer/blob/main/scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh
            $d_model = 256
            $d_ff = 256
            
            declare -A d_model_and_ff_tune
            d_model_and_ff_tune[96] = 256
            d_model_and_ff_tune[192] = 256
            d_model_and_ff_tune[336] = 512
            d_model_and_ff_tune[720] = 512

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
              --pred_len $pred_len \
              --e_layers 2 \
              --enc_in 7 \
              --dec_in 7 \
              --c_out 7 \
              --des 'Exp' \
              --d_model ${d_model_and_ff_tune[$pred_len]} \
              --d_ff ${d_model_and_ff_tune[$pred_len]} \
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
