import glob
import os
from config_utils import *
import copy

BASE_CONFIGS_FILE = "exp_configs.json"
OUTPUT_CONFIGS_FILE = "generated_exp_configs.json"


base_configs = load_configs_from_yaml(BASE_CONFIGS_FILE)

datasets_path_mapping = {
    "ETTh1" : "./dataset/ETT-small/ETTh1/",
    "ETTh2" : "./dataset/ETT-small/ETTh2/",
    "ETTm1" : "./dataset/ETT-small/ETTm1/",
    "ETTm2" : "./dataset/ETT-small/ETTm2/",
    "ECL": "./dataset/electricity/ECL/",
    "Traffic": "./dataset/traffic/Traffic/",
    "Weather": "./dataset/weather/Weather/"
}


list_of_configs = []
list_of_config_hashes = []
# Get a list of all CSV files in the specified folder
prediction_lengths = [96, 192, 336, 720]
models = ["iTransformer", "TimesNet", "TimeXer", "PatchTST", "Nonstationary_Transformer", "Crossformer", "Autoformer"]

found = {}
for dataset_name in datasets_path_mapping.keys():
    for prediction_length in prediction_lengths:
        found[f"{dataset_name}_{prediction_length}"] = []

shaped_datasets_count = {dataset : 0 for dataset in datasets_path_mapping.keys()}

base_configs_count = 0

for model in models:
    for pred_len in prediction_lengths:
        for dataset_name, dataset_root_path in datasets_path_mapping.items(): # ABLATION FOR BASE DATASETS
            csv_files = glob.glob(os.path.join(dataset_root_path, '*.csv'))
            for config in base_configs: # SEARCHING MODELS AND PREDICTION LENGTHS IN BASE CONFIGS
                if config.model == model and config.forecast.pred_len == pred_len and config.data.name == dataset_name: 
                    base_configs_count += 1
                    found[f"{dataset_name}_{config.forecast.pred_len}"] += [config.model]
                    shaped_datasets_count[dataset_name] = len(csv_files)
                    
                    for file_path in csv_files: # ABLATION FOR SHAPED DATASETS
                        config_copy = copy.deepcopy(config)
                        full_relative_path = dataset_root_path
                        file_name = os.path.basename(file_path)
                    
                        config_copy.data.root_path = full_relative_path
                        config_copy.data.data_path = file_name
                        config_copy.model_id = f"{pred_len}_96{file_name}"

                        config_copy.data.features = "MS"

                        config_hash = compute_config_hash(config_copy) # ensuring non duplicates
                        config_copy.experiment_id = config_hash
                        if config_hash not in list_of_config_hashes:
                            list_of_configs += [config_copy]
                            list_of_config_hashes += [config_hash]
                
print(f"base_configs_count = {base_configs_count}")

print(f"Number of experiments configurations generated: {len(list_of_configs)}")
print(f"Expected number of experiments configurations to be generated: {len(list_of_configs)}")

print(f"Model Vs Pred_Len found: {[(found_key,len(found_value)) for found_key, found_value in found.items()]}")
print(f"Found: {found}")

print(f"Shaped_datasets_count: {shaped_datasets_count}")

save_configs_to_json(list_of_configs, output_file=OUTPUT_CONFIGS_FILE)