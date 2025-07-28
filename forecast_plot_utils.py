import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"
import os
from config_utils import datasets_path_mapping, datasets_split_mapping
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def find_folders_by_partial_name(root_dir, partial_name):
    matching_folders = []
    for entry in os.listdir(root_dir):
        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path) and (partial_name in entry):
            matching_folders.append(entry)
    return matching_folders


def get_forecast_plot(
    pwindow=501,
    pconstructor="ipfarm",
    dataset="ECL",
    pred_len=96,
    model="Autoformer",
    offset_idx=0,
    root_directory="./results",
    with_saliency_ratio=False,
    with_min_max_scaling=False
):

    dataset_modified_name = dataset if dataset != "Traffic" else "TrafficL"

    root_path_partial_name = f'{pred_len}_96{dataset_modified_name}_w{pwindow}_{pconstructor}.csv_{model}_{dataset}'
    root_path_partial_name_identity = f'{pred_len}_96{dataset_modified_name}.csv_{model}_{dataset}'

    matched_folders = find_folders_by_partial_name(root_directory, root_path_partial_name)
    matched_folders_identity = find_folders_by_partial_name(root_directory, root_path_partial_name_identity)

    if len(matched_folders) > 1:
        raise ValueError(f"Multiple folders found with partial_name '{root_path_partial_name}': {matched_folders}")
    elif len(matched_folders) == 0:
        raise FileNotFoundError(f"No folders found with partial_name '{root_path_partial_name}' in {root_directory}")

    if len(matched_folders_identity) > 1:
        raise ValueError(f"Multiple folders found with identity partial_name '{root_path_partial_name_identity}': {matched_folders_identity}")
    elif len(matched_folders_identity) == 0:
        raise FileNotFoundError(f"No folders found with identity partial_name '{root_path_partial_name_identity}' in {root_directory}")

    root_path = matched_folders[0]
    root_path_identity = matched_folders_identity[0]

    # shaped prediction
    data_pred = np.load(f'{root_directory}/{root_path}/pred.npy')
    data_pred = data_pred.squeeze()
    df_pred = pd.DataFrame(data_pred)

    data_true = np.load(f'{root_directory}/{root_path}/true.npy')
    data_true = data_true.squeeze()
    df_true = pd.DataFrame(data_true)

    # identity prediction
    data_pred_identity = np.load(f'{root_directory}/{root_path_identity}/pred.npy')
    data_pred_identity = data_pred_identity.squeeze()
    df_pred_identity = pd.DataFrame(data_pred_identity)

    data_true_identity = np.load(f'{root_directory}/{root_path_identity}/true.npy')
    data_true_identity = data_true_identity.squeeze()
    df_true_identity = pd.DataFrame(data_true_identity)

    seq_len = 96

    shift_map = {
        "ETTh1": 0,
        "ETTh2": 0,
        "ETTm1": 0,
        "ETTm2": 0,
        "Weather": -3,
        "ECL": -2,
        "Traffic": -2,
    }

    # NOTE: for some reason, it seems the indexes of df_dataset and df_dataset_identity are not matching
    # Need further investigation
    # Fixing by hard shifting the necessary amount

    shift = shift_map.get(dataset, 0)

    test_start_index = datasets_split_mapping[dataset][0] + datasets_split_mapping[dataset][1] - 1 + seq_len - 1
    test_end_index = datasets_split_mapping[dataset][0] + datasets_split_mapping[dataset][1] - 1  + datasets_split_mapping[dataset][2] - 1 + seq_len - 1


    # FORECAST PLOT
    ts_true = df_true.iloc[offset_idx]
    ts_pred = df_pred.iloc[offset_idx]

    ts_true_identity = df_true_identity.iloc[offset_idx]
    ts_pred_identity = df_pred_identity.iloc[offset_idx]

    forecast_plot = pd.DataFrame({
        "true": ts_true,
        "pred": ts_pred,
        "true_identity": ts_true_identity,
        "pred_identity": ts_pred_identity,
    })


    # DATASET PLOT

    train_size = datasets_split_mapping[dataset][0] # for scaling purposes

    dataset_file_name = f"{dataset_modified_name}_w{pwindow}_{pconstructor}.csv"
    dataset_file_name_identity = f"{dataset_modified_name}.csv"

    dataset_full_path = f"{datasets_path_mapping[dataset]}/{dataset_file_name}"
    dataset_full_path_identity = f"{datasets_path_mapping[dataset]}/{dataset_file_name_identity}"

    df_dataset = pd.read_csv(dataset_full_path)
    df_dataset = df_dataset.set_index("date")
    scaler_dataset = StandardScaler().set_output(transform="pandas")
    scaler_dataset.fit(df_dataset.iloc[:train_size]) # scaling based only on training data as in forecasting/training
    df_dataset = scaler_dataset.transform(df_dataset)

    df_dataset = df_dataset.reset_index().drop_duplicates(subset=['date'], keep="first").set_index('date') # Weather dataset contains duplicate at 2020-05-12 06:00:00

    df_dataset_identity = pd.read_csv(dataset_full_path_identity)
    df_dataset_identity = df_dataset_identity.set_index("date")
    scaler_dataset_identity = StandardScaler().set_output(transform="pandas")
    scaler_dataset_identity.fit(df_dataset_identity.iloc[:train_size]) # scaling based only on training data as in forecasting/training
    df_dataset_identity = scaler_dataset_identity.transform(df_dataset_identity)

    df_dataset_identity = df_dataset_identity.reset_index().drop_duplicates(subset=['date'], keep="first").set_index('date')  # Weather dataset contains duplicate at 2020-05-12 06:00:00

    if with_saliency_ratio:
        columns_to_calculate_ratio = [column for column in df_dataset.columns if "OT" not in column]
        columns_to_calculate_ratio_identity = [column for column in df_dataset_identity.columns if "OT" not in column]

        dataset_plot = df_dataset[columns_to_calculate_ratio].div(df_dataset_identity[columns_to_calculate_ratio_identity])

        dataset_plot.columns = [f"{col}_ratio" for col in columns_to_calculate_ratio]
        dataset_plot["OT"] = df_dataset["OT"]
    else:
        dataset_plot = pd.merge(df_dataset, df_dataset_identity, left_index=True, right_index=True, suffixes=('_shaped', '_identity'))


    dataset_plot = dataset_plot.iloc[test_start_index+shift:test_end_index+shift]

    idx_min = offset_idx
    idx_max = idx_min + pred_len

    dataset_plot = dataset_plot.iloc[idx_min:idx_max]

    # JOINT DATAFRAME

    with_min_max_scaling = False
    scaler_min_max = MinMaxScaler().set_output(transform="pandas")

    forecast_plot.index = dataset_plot.index

    joint_plot = pd.merge(dataset_plot, forecast_plot, left_index=True, right_index=True, suffixes=('_dataset', '_forecast'))

    if with_min_max_scaling:
        joint_plot = scaler_min_max.fit_transform(joint_plot)


    # RETURN PLOTS
    return forecast_plot, dataset_plot, joint_plot

    