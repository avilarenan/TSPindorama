import glob
import os
from config_utils import *
import copy


# NOTE: the following script expects a base experiment config file and extends it with ablation combinations for the prediction length and dataset parameter
# TODO: make the ablation for models list by getting the base experiment config file in a search

exp_configs = load_configs_from_yaml("ETT_exp_test.json")

folder_path = "./dataset/ETT-small/"  # Replace with the actual path
list_of_configs = []
# Get a list of all CSV files in the specified folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

prediction_lengths = [96, 192, 336, 720]

for config in exp_configs:
    for file_path in csv_files: # ABLATION FOR DATASETS
        config_copy = copy.deepcopy(config)
        # Get the full path
        full_relative_path = file_path
        
        # Get the filename (including extension)
        file_name = os.path.basename(file_path)
        
        print(f"Full Path: {full_relative_path}")
        print(f"File Name: {file_name}")
        print("-" * 20)
        config_copy.data.root_path = full_relative_path
        config_copy.data.data_path = file_name

        for prediction_length in prediction_lengths:
            config_copy_nested = copy.deepcopy(config_copy)
            config_copy_nested.forecast.pred_len = prediction_length
            
            list_of_configs += [config_copy_nested]

save_configs_to_json(list_of_configs, output_file="generated_exp_configs.json")