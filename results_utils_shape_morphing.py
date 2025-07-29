import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import stats, interpolate
import os

pd.options.plotting.backend = "plotly"
from config_utils import datasets_path_mapping, datasets_split_mapping
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_results(with_calc_improv=True, results_directory="./results"):
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

            try:
                split_index = dir_name.index("_w")
                split_index_last = dir_name.index(".csv")
                pwindow = int(dir_name[split_index+1:].split("_")[0][1:])
                pwindow_str_len = len(f"{pwindow}")
                pconstructor = dir_name[split_index+3+pwindow_str_len:split_index_last+4]
            except Exception as e:
                pwindow = None
                if "duplicated" in dir_name:
                    pconstructor = "duplicated"
                elif "zeroed" in dir_name:
                    pconstructor = "zeroed"
                else:
                    pconstructor = "identity"
            
            try:
                pred_len = dir_name.split("long_term_forecast")[1].split("_")[2]
                df["pred_len"] = int(pred_len)
            except Exception as e:
                pred_len = dir_name.split("long_term_forecast")[1].split("_")[1]
                df["pred_len"] = int(pred_len)

            try:
                lookback_len = dir_name.split("long_term_forecast")[1].split("_")[3]
                lookback_len_cleaned = ""
                for character in lookback_len:
                    if character.isdigit():
                        lookback_len_cleaned += character
                    else:
                        break
                df["lookback_len"] = int(lookback_len_cleaned)
            except Exception as e:
                lookback_len = dir_name.split("long_term_forecast")[1].split("_")[2]
                lookback_len_cleaned = ""
                for character in lookback_len:
                    if character.isdigit():
                        lookback_len_cleaned += character
                    else:
                        break
                df["lookback_len"] = int(lookback_len_cleaned)
            

            df["pwindow"] = pwindow
            df["pconstructor"] = pconstructor
            dataset_name = dir_name[dir_name.find("96ETTh1")+7: dir_name.find("_ft")].split("_")[-1]
            if dataset_name == "Traffic":
                dataset_name = "TrafficL"
            df["dataset"] = dataset_name
            list_of_metrics_dfs += [df]

    results_df = pd.concat(list_of_metrics_dfs).sort_values(by="mse", ascending=True)

    if with_calc_improv is False:
        return results_df

    # CALCULATING improvement
    df = results_df

    identity_df = df[df["pconstructor"] == "identity"].drop(["pconstructor", "pwindow"], axis=1).set_index(["model_name", "pred_len", "dataset"])

    df_results_indexed = results_df.set_index(["model_name", "pred_len", "dataset"])

    merged_df_inner = pd.merge(df_results_indexed, identity_df, left_index=True, right_index=True, how='inner', suffixes=["", "_identity"])
    merged_df_inner["mse_diff"] = merged_df_inner["mse_identity"] - merged_df_inner["mse"]
    merged_df_inner["improvement"] = merged_df_inner["mse_diff"]/merged_df_inner["mse_identity"] * 100
    merged_df_inner = merged_df_inner.reset_index()

    return merged_df_inner

def plot_results(
    df: pd.DataFrame,
    value_col: str,
    group_col: str | list[str] | None = None,
    bins: int | str | list[int] = "auto",
    smooth: str = "kde",                # "kde", "interp", or None
    kde_bw: float | str | None = None,
    backend: str = "matplotlib",        # "matplotlib" or "plotly"
    figsize=(8, 5),
    hist_kwargs=None,
    line_kwargs=None,
    vline_at_zero: bool = False,
    vline_at_peak: bool = False,
    min_peak_x: float | None = None,
    y_agg_col: str | None = None,       # y-axis: aggregate this column instead of count
    agg_func=np.mean                    # aggregation function (e.g., np.mean, np.sum)
):
    hist_kwargs = hist_kwargs or {}
    line_kwargs = line_kwargs or {}

    if group_col is None:
        df = df.assign(_tmp_group="all")
        group_labels = ["_tmp_group"]
    elif isinstance(group_col, str):
        group_labels = [group_col]
    else:
        df = df.copy()
        df["_tmp_group"] = df[group_col].astype(str).agg(" / ".join, axis=1)
        group_labels = ["_tmp_group"]

    group_col = group_labels[0]
    groups = df[group_col].dropna().unique()
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    bin_edges = None
    if isinstance(bins, int):
        bin_edges = np.histogram_bin_edges(df[value_col].dropna(), bins=bins)
    elif isinstance(bins, (list, np.ndarray)):
        bin_edges = np.array(bins)

    if backend == "matplotlib":
        fig, ax = plt.subplots(figsize=figsize)

        for i, g in enumerate(groups):
            sub = df[df[group_col] == g]
            values = sub[value_col].dropna()

            if y_agg_col is None:
                counts, _ = np.histogram(values, bins=bin_edges)
                x_mid = (bin_edges[:-1] + bin_edges[1:]) / 2
                y_vals = counts
            else:
                sub = sub[[value_col, y_agg_col]].dropna()
                sub["bin"] = pd.cut(sub[value_col], bins=bin_edges, labels=False, include_lowest=True)
                agg = sub.groupby("bin")[y_agg_col].agg(agg_func)
                y_vals = agg.values
                x_mid = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in agg.index]

            if len(x_mid) < 3:
                continue

            if smooth == "kde":
                if y_agg_col is not None:
                    raise ValueError("KDE smoothing is not supported when using y_agg_col.")
                kde = stats.gaussian_kde(values, bw_method=kde_bw)
                x_dense = np.linspace(min(x_mid), max(x_mid), 500)
                y_dense = kde(x_dense) * len(values) * np.diff(bin_edges)[0]
                x_peak = x_dense[np.argmax(y_dense)]

                if min_peak_x is not None and x_peak < min_peak_x:
                    continue

                ax.plot(x_dense, y_dense, label=f"{g} (KDE)", color=palette[i % len(palette)], **line_kwargs)

                if vline_at_peak:
                    ax.axvline(x_peak, color=palette[i % len(palette)], linestyle=":", linewidth=1)
                    ax.annotate(f"{x_peak:.2f}", xy=(x_peak, max(y_dense)), xytext=(0, 5),
                                textcoords="offset points", ha="center", fontsize=8,
                                color=palette[i % len(palette)])

            elif smooth == "interp":
                x_dense = np.linspace(min(x_mid), max(x_mid), 500)
                spline = interpolate.make_interp_spline(x_mid, y_vals, k=min(3, len(x_mid)-1))
                y_dense = spline(x_dense)
                x_peak = x_dense[np.argmax(y_dense)]

                if min_peak_x is not None and x_peak < min_peak_x:
                    continue

                ax.plot(x_dense, y_dense, label=f"{g} (spline)", color=palette[i % len(palette)], **line_kwargs)

                if vline_at_peak:
                    ax.axvline(x_peak, color=palette[i % len(palette)], linestyle=":", linewidth=1)
                    ax.annotate(f"{x_peak:.2f}", xy=(x_peak, max(y_dense)), xytext=(0, 5),
                                textcoords="offset points", ha="center", fontsize=8,
                                color=palette[i % len(palette)])

            # Plot bars regardless of y_agg_col
            bar_label = f"{g} (hist)" if y_agg_col is None else f"{g} (bars)"
            ax.bar(x_mid, y_vals, width=np.diff(bin_edges), alpha=0.4,
                   color=palette[i % len(palette)], label=bar_label, **hist_kwargs)

        if vline_at_zero:
            ax.axvline(0, color="black", linestyle="--", linewidth=1)

        ax.set_xlabel(value_col)
        ax.set_ylabel("count" if y_agg_col is None else f"{agg_func.__name__}({y_agg_col})")
        ax.set_title(f"Grouped by {group_col}")
        ax.legend()
        fig.tight_layout()
        plt.show()

    elif backend == "plotly":
        fig = go.Figure()

        for i, g in enumerate(groups):
            sub = df[df[group_col] == g]
            values = sub[value_col].dropna()

            if y_agg_col is None:
                counts, _ = np.histogram(values, bins=bin_edges)
                x_mid = (bin_edges[:-1] + bin_edges[1:]) / 2
                y_vals = counts
            else:
                sub = sub[[value_col, y_agg_col]].dropna()
                sub["bin"] = pd.cut(sub[value_col], bins=bin_edges, labels=False, include_lowest=True)
                agg = sub.groupby("bin")[y_agg_col].agg(agg_func)
                y_vals = agg.values
                x_mid = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in agg.index]

            if len(x_mid) < 3:
                continue

            if smooth == "kde":
                if y_agg_col is not None:
                    raise ValueError("KDE smoothing is not supported when using y_agg_col.")
                kde = stats.gaussian_kde(values, bw_method=kde_bw)
                x_dense = np.linspace(min(x_mid), max(x_mid), 500)
                y_dense = kde(x_dense) * len(values) * np.diff(bin_edges)[0]
                x_peak = x_dense[np.argmax(y_dense)]

                if min_peak_x is not None and x_peak < min_peak_x:
                    continue

                fig.add_trace(go.Scatter(
                    x=x_dense, y=y_dense, mode="lines",
                    name=f"{g} (KDE)",
                    line=dict(color=palette[i % len(palette)], **line_kwargs)
                ))

                if vline_at_peak:
                    fig.add_vline(
                        x=x_peak, line_dash="dot",
                        line_color=palette[i % len(palette)],
                        line_width=1,
                        annotation_text=f"{x_peak:.2f}",
                        annotation_position="top",
                        annotation_font=dict(size=10, color=palette[i % len(palette)])
                    )

            elif smooth == "interp":
                x_dense = np.linspace(min(x_mid), max(x_mid), 500)
                spline = interpolate.make_interp_spline(x_mid, y_vals, k=min(3, len(x_mid)-1))
                y_dense = spline(x_dense)
                x_peak = x_dense[np.argmax(y_dense)]

                if min_peak_x is not None and x_peak < min_peak_x:
                    continue

                fig.add_trace(go.Scatter(
                    x=x_dense, y=y_dense, mode="lines",
                    name=f"{g} (spline)",
                    line=dict(color=palette[i % len(palette)], **line_kwargs)
                ))

                if vline_at_peak:
                    fig.add_vline(
                        x=x_peak, line_dash="dot",
                        line_color=palette[i % len(palette)],
                        line_width=1,
                        annotation_text=f"{x_peak:.2f}",
                        annotation_position="top",
                        annotation_font=dict(size=10, color=palette[i % len(palette)])
                    )

            # Plot bars regardless of y_agg_col
            bar_label = f"{g} (hist)" if y_agg_col is None else f"{g} (bars)"
            fig.add_trace(go.Bar(
                x=x_mid,
                y=y_vals,
                name=bar_label,
                opacity=0.5,
                marker_color=palette[i % len(palette)],
                **hist_kwargs
            ))

        if vline_at_zero:
            fig.add_vline(
                x=0, line_dash="dash", line_color="black",
                line_width=1, annotation_text="0",
                annotation_position="top left"
            )

        fig.update_layout(
            title=f"Grouped by {group_col}",
            xaxis_title=value_col,
            yaxis_title="count" if y_agg_col is None else f"{agg_func.__name__}({y_agg_col})",
            barmode="overlay",
        )
        fig.show()

    else:
        raise ValueError("backend must be 'matplotlib' or 'plotly'")


def get_best_results(results_df):
    tmp_results_df = results_df.copy()
    tmp_results_df = tmp_results_df[tmp_results_df["pconstructor"] != "identity"]
    idx = tmp_results_df.groupby(["dataset", "pred_len", "model_name"])['mse'].idxmin()
    return tmp_results_df.loc[idx].sort_values(by=["dataset", "mse"], ascending=True)

def shape_best_result_for_final_format_compact(best_results_df, include_improvement_in_meta=False):
    df = best_results_df.copy()
    df['pconstructor'] = df['pconstructor'].str.replace('.csv', '', regex=False)

    # Updated formatting with scientific notation for MSEs
    def format_perf(row):
        try:
            mse_sci = f"{row['mse']:.3e}"
            mse_id_sci = f"{row['mse_identity']:.3e}"
            delta = f"{row['improvement']:.1f}%"
            if row['improvement'] < 0:
                delta = f"{row['improvement']:.1f}%↓"
            if row['improvement'] > 0:
                delta = f"{row['improvement']:.1f}%↑"
            return f"{mse_sci}/{mse_id_sci}({delta})"
        except:
            return ""

    df['perf'] = df.apply(format_perf, axis=1)

    # Performance table
    perf_df = df.pivot_table(
        index=['dataset', 'pred_len'],
        columns='model_name',
        values='perf',
        aggfunc='first'
    ).sort_index(axis=1)

    # Metadata table
    meta_cols = ['pwindow', 'pconstructor']
    if include_improvement_in_meta:
        meta_cols.append('improvement')

    meta_df = df.pivot_table(
        index=['dataset', 'pred_len'],
        columns='model_name',
        values=meta_cols,
        aggfunc='first'
    ).sort_index(axis=1, level=0)

    return perf_df, meta_df


def get_forecast_plot(model_name, dataset, pred_len, cutoff_index, best_results, all_results):

    exp_name = best_results[
        (best_results["model_name"] == model_name) & 
        (best_results["dataset"] == dataset) & 
        (best_results["pred_len"] == pred_len)
    ]["exp_name"].values[0] # there should be only one with this filter
    print(exp_name)

    exp_name_identity = all_results[
        (all_results["model_name"] == model_name) & 
        (all_results["dataset"] == dataset) & 
        (all_results["pred_len"] == pred_len) &
        (all_results["pconstructor"] == "identity")
    ]["exp_name"].values[0] # there should be only one with this filter
    print(exp_name_identity)

    metrics_shaped = pd.read_csv(f"./results/{exp_name}/_metrics.csv")
    print(f"Metrics shaped: {metrics_shaped}")
    metrics_shaped = pd.read_csv(f"./results/{exp_name_identity}/_metrics.csv")
    print(f"Metrics identity: {metrics_shaped}")

    pred_df = pd.read_csv(f"./results/{exp_name}/preds_vs_trues.csv")
    pred_df_identity = pd.read_csv(f"./results/{exp_name_identity}/preds_vs_trues.csv")

    p = pred_df[pred_df["cutoff_index"]==cutoff_index]
    p_i = pred_df_identity[pred_df_identity["cutoff_index"]==cutoff_index]

    p = p.rename({"preds": "preds_shaped", "true": "true_shaped"}, axis=1)
    p_i = p_i.rename({"preds": "preds_identity", "true": "true_identity"}, axis=1)

    ret_df = pd.concat([p, p_i], axis=1)[["preds_shaped", "preds_identity", "true_shaped"]]
    return ret_df.rename({"true_shaped": "true"}, axis=1)


def find_folders_by_partial_name(root_dir, partial_name):
    matching_folders = []
    for entry in os.listdir(root_dir):
        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path) and (partial_name in entry):
            matching_folders.append(entry)
    return matching_folders


def get_forecast_shape_morphing_plot(
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

    