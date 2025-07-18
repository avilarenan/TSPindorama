import pandas as pd

def get_best_hyperparam_mapping():
    """
    Reads the best results from a CSV file and constructs a mapping of datasets to models and their best hyperparameters.
    
    Returns:
        dict: A nested dictionary mapping datasets to models and their best hyperparameters.
        The structure is:
        {       
            "dataset_name": {
                "model_name": {
                    "pconstructor": best_pconstructor_value,
                    "pwindow": best_pwindow_value
                }
            }
        }
    """
    constructor_window_mapping = {}
    
    # Load the CSV file
    file_path = "./best_results_summarized.csv" 
    df = pd.read_csv(file_path)

    # Keep only the best configuration (max improvement) per (dataset, model_name)
    best_configs = df.loc[df.groupby(["dataset", "model_name"])["improvement"].idxmax()]

    # Build the nested dictionary
    for _, row in best_configs.iterrows():
        dataset = row["dataset"]
        model = row["model_name"]
        pconstructor = row["pconstructor"]
        pwindow = int(row["pwindow"])

        if dataset not in constructor_window_mapping:
            constructor_window_mapping[dataset] = {}

        constructor_window_mapping[dataset][model] = {
            "pconstructor": pconstructor,
            "pwindow": pwindow
        }
    
    return constructor_window_mapping