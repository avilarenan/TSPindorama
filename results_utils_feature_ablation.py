import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import stats, interpolate
import os

def get_results(results_directory="./results"):
    list_of_metrics_dfs = []
    directory_path = results_directory
    for root, dirs, files in os.walk(directory_path):
        for dir_name in dirs:
            df = pd.read_csv(f"{os.path.join(root, dir_name)}/_metrics.csv")

            df["exp_name"] = dir_name

            # example1: long_term_forecast_ETTh1_96_96ETTh1_TiDE_ETTh1_ftMS_sl96_ll48_pl96_dm256_nh8_el2_dl2_df256_expand2_dc4_fc1_ebtimeF_dtTrue_test_0 
            # example2: long_term_forecast_ETTh1_96_96ETTh1_w501_ipentropy_TiDE_ETTh1_ftMS_sl96_ll48_pl96_dm256_nh8_el2_dl2_df256_expand2_dc4_fc1_ebtimeF_dtTrue_test_0
            
            split_index = dir_name.index("_ft")
            model_name = dir_name[:split_index].split("_")[-2]
            df["model_name"] = model_name


            start_split_word = "powerset"
            end_split_word = ".csv"
            powerset_split = dir_name.index(start_split_word)
            csv_split = dir_name.index(end_split_word)
            features_powerset = dir_name[powerset_split+len(start_split_word)+1:csv_split]
            df["features_powerset"] = f"{features_powerset.split("_")}"

            pred_len = dir_name[:powerset_split].split("_")[3]
            df["pred_len"] = pred_len

            dataset_name = dir_name[dir_name.find("96ETTh1")+7: dir_name.find("_ft")].split("_")[-1]
            if dataset_name == "Traffic":
                dataset_name = "TrafficL"
            df["dataset"] = dataset_name
            list_of_metrics_dfs += [df]

    results_df = pd.concat(list_of_metrics_dfs).sort_values(by="mse", ascending=True)


    return results_df


def find_folders_with_include_exclude(root_dir, required_words=None, forbidden_words=None):
    """
    Find folders that contain all required words and none of the forbidden words.
    
    Args:
        root_dir (str): Directory to scan.
        required_words (list of str): Words that MUST appear in the folder name.
        forbidden_words (list of str): Words that MUST NOT appear in the folder name.

    Returns:
        List of matching folder names (not full paths).
    """
    required_words = required_words or []
    forbidden_words = forbidden_words or []
    matching_folders = []

    for entry in os.listdir(root_dir):
        full_path = os.path.join(root_dir, entry)

        if not os.path.isdir(full_path):
            continue

        # Check all required words are in the name
        if not all(word in entry for word in required_words):
            continue

        # Check none of the forbidden words are in the name
        if any(word in entry for word in forbidden_words):
            continue

        matching_folders.append(entry)

    return matching_folders


def get_forecast_plot(dataset_name, model, pred_len, features_of_interest, root_directory, offset_idx=0):

    exogenous_features_map = {
        "ETTh1": ["HULL", "HUFL", "LULL", "LUFL", "MULL", "MUFL"],
        "EPEX-DE": [],
        "EnergyLoad": []
    }

    exogenous_features = exogenous_features_map[dataset_name]
    forbidden_words = list(set(exogenous_features) - set(features_of_interest))
    words_filter = [dataset_name] + [model] + features_of_interest + [f"_{pred_len}_"]

    matched_folders = find_folders_with_include_exclude(root_directory, required_words=words_filter, forbidden_words=forbidden_words)

    if len(matched_folders) > 1:
        raise ValueError(f"Multiple folders found with words_filter '{words_filter}': {matched_folders}")
    elif len(matched_folders) == 0:
        raise FileNotFoundError(f"No folders found with words_filter '{words_filter}' in {root_directory}")

    root_path = matched_folders[0]

    # shaped prediction
    data_pred = np.load(f'{root_directory}/{root_path}/pred.npy')
    data_pred = data_pred.squeeze()
    df_pred = pd.DataFrame(data_pred)

    data_true = np.load(f'{root_directory}/{root_path}/true.npy')
    data_true = data_true.squeeze()
    df_true = pd.DataFrame(data_true)

    offset_idx = 0

    # FORECAST PLOT
    ts_true = df_true.iloc[offset_idx]
    ts_pred = df_pred.iloc[offset_idx]

    forecast_plot = pd.DataFrame({
        "true": ts_true,
        "pred": ts_pred
    })

    return forecast_plot