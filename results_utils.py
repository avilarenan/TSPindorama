import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import stats, interpolate
import os

def plot_grouped_hist_with_curve(
    df: pd.DataFrame,
    value_col: str,
    group_col: str | list[str] | None = None,
    bins: int | str | list[int] = "auto",
    smooth: str = "kde",               # "kde", "interp", or None
    kde_bw: float | str | None = None,
    backend: str = "matplotlib",       # "matplotlib" or "plotly"
    figsize=(8, 5),
    hist_kwargs=None,
    line_kwargs=None,
    vline_at_zero: bool = False,
    vline_at_peak: bool = False,
    min_peak_x: float | None = None    # ← NEW: skip curves with x_peak below this
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

    if backend == "matplotlib":
        fig, ax = plt.subplots(figsize=figsize)

        for i, g in enumerate(groups):
            sub = df[df[group_col] == g][value_col].dropna()
            counts, bin_edges, _ = np.histogram(sub, bins=bins)

            if smooth:
                x_mid = (bin_edges[:-1] + bin_edges[1:]) / 2

                if smooth == "kde":
                    kde = stats.gaussian_kde(sub, bw_method=kde_bw)
                    x_dense = np.linspace(bin_edges.min(), bin_edges.max(), 500)
                    y_dense = kde(x_dense) * len(sub) * np.diff(bin_edges)[0]
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
                    spline = interpolate.make_interp_spline(x_mid, counts, k=min(3, len(x_mid) - 1))
                    x_dense = np.linspace(x_mid.min(), x_mid.max(), 500)
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

            # Show histogram only if curve is shown
            ax.hist(sub, bins=bins, alpha=0.4, density=False,
                    color=palette[i % len(palette)], label=f"{g} (hist)", **hist_kwargs)

        if vline_at_zero:
            ax.axvline(0, color="black", linestyle="--", linewidth=1)

        ax.set_xlabel(value_col)
        ax.set_ylabel("count")
        ax.set_title(f"Histogram of '{value_col}' grouped by {group_col}")
        ax.legend()
        fig.tight_layout()
        plt.show()

    elif backend == "plotly":
        fig = go.Figure()

        for i, g in enumerate(groups):
            sub = df[df[group_col] == g][value_col].dropna()
            counts, bin_edges = np.histogram(sub, bins=bins)
            x_mid = (bin_edges[:-1] + bin_edges[1:]) / 2

            if smooth == "kde":
                kde = stats.gaussian_kde(sub, bw_method=kde_bw)
                x_dense = np.linspace(bin_edges.min(), bin_edges.max(), 500)
                y_dense = kde(x_dense) * len(sub) * np.diff(bin_edges)[0]
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
                spline = interpolate.make_interp_spline(x_mid, counts, k=min(3, len(x_mid) - 1))
                x_dense = np.linspace(x_mid.min(), x_mid.max(), 500)
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

            fig.add_histogram(
                x=sub, name=f"{g} (hist)",
                opacity=0.5,
                nbinsx=bins if isinstance(bins, int) else None,
                marker_color=palette[i % len(palette)],
                **hist_kwargs
            )

        if vline_at_zero:
            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="black",
                line_width=1,
                annotation_text="0",
                annotation_position="top left"
            )

        fig.update_layout(
            title=f"Histogram of '{value_col}' grouped by {group_col}",
            xaxis_title=value_col,
            yaxis_title="count",
            barmode="overlay",
        )
        fig.show()

    else:
        raise ValueError("backend must be 'matplotlib' or 'plotly'")

def get_results():


    list_of_metrics_dfs = []
    directory_path = "./results"
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
                pwindow = int(dir_name[split_index+1:].split("_")[0][1:])
                pconstructor = dir_name[split_index+1:].split("_")[1]
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

            df["dataset"] = dir_name[dir_name.find("96ETTh1")+7: dir_name.find("_ft")].split("_")[-1]

            list_of_metrics_dfs += [df]

    results_df = pd.concat(list_of_metrics_dfs).sort_values(by="mse", ascending=True)

    # CALCULATING improvement
    df = results_df

    identity_df = df[df["pconstructor"] == "identity"].drop(["pconstructor", "pwindow"], axis=1).set_index(["model_name", "pred_len", "dataset"])

    df_results_indexed = results_df.set_index(["model_name", "pred_len", "dataset"])

    merged_df_inner = pd.merge(df_results_indexed, identity_df, left_index=True, right_index=True, how='inner', suffixes=["", "_identity"])
    merged_df_inner["mse_diff"] = merged_df_inner["mse_identity"] - merged_df_inner["mse"]
    merged_df_inner["improvement"] = merged_df_inner["mse_diff"]/merged_df_inner["mse_identity"] * 100
    merged_df_inner = merged_df_inner.reset_index()

    return merged_df_inner

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import stats, interpolate

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

            if y_agg_col is None:
                ax.hist(values, bins=bin_edges, alpha=0.4, density=False,
                        color=palette[i % len(palette)], label=f"{g} (hist)", **hist_kwargs)

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

            if y_agg_col is None:
                fig.add_histogram(
                    x=values, name=f"{g} (hist)",
                    opacity=0.5,
                    nbinsx=len(bin_edges)-1 if bin_edges is not None else bins,
                    marker_color=palette[i % len(palette)],
                    **hist_kwargs
                )

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
