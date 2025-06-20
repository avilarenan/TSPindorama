import os
import torch
import torch.backends
from exp.exp_long_term_forecasting_br import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np

import torch
import torch.backends
import random
import numpy as np

from exp.exp_long_term_forecasting_br import Exp_Long_Term_Forecast # modernized
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification

from exp_params import ExperimentConfig
from config_utils import *

def run_experiment(args):
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Setup device
    if torch.cuda.is_available() and args.gpu.use_gpu:
        device = torch.device(f'cuda:{args.gpu.gpu}')
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            device = torch.device("cpu")
        print('Using cpu or mps')

    args.device = device

    if args.gpu.use_gpu and args.gpu.use_multi_gpu:
        device_ids = [int(id_) for id_ in args.gpu.devices.replace(' ', '').split(',')]
        args.device_ids = device_ids
        args.gpu.gpu = device_ids[0]

    print('Args in experiment:')
    print(args)

    # Select the appropriate experiment class
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        raise Exception(f"short_term_forecast task not implemented yet.")
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        raise Exception(f"imputation task not implemented yet.")
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        raise Exception(f"anomaly_detection task not implemented yet.")
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        raise Exception(f"classification task not implemented yet.")
        Exp = Exp_Classification
    else:
        raise Exception(f"Unexpected task_name {args.task_name}.")

    if args.is_training:
        for ii in range(args.optimization.itr):
            exp = Exp(args)

            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data.name}_ft{args.data.features}_' \
                      f'sl{args.forecast.seq_len}_ll{args.forecast.label_len}_pl{args.forecast.pred_len}_' \
                      f'dm{args.model_params.d_model}_nh{args.model_params.n_heads}_el{args.model_params.e_layers}_' \
                      f'dl{args.model_params.d_layers}_df{args.model_params.d_ff}_' \
                      f'expand{args.model_params.expand}_dc{args.model_params.d_conv}_' \
                      f'fc{args.model_params.factor}_eb{args.model_params.embed}_dt{args.model_params.distil}_' \
                      f'{args.des}_{ii}'

            print(f'>>>>>>> Start training: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>> Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)

            if args.gpu.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu.gpu_type == 'cuda':
                torch.cuda.empty_cache()

    else:
        exp = Exp(args)
        ii = 0
        setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data.name}_ft{args.data.features}_' \
                  f'sl{args.forecast.seq_len}_ll{args.forecast.label_len}_pl{args.forecast.pred_len}_' \
                  f'dm{args.model_params.d_model}_nh{args.model_params.n_heads}_el{args.model_params.e_layers}_' \
                  f'dl{args.model_params.d_layers}_df{args.model_params.d_ff}_' \
                  f'expand{args.model_params.expand}_dc{args.model_params.d_conv}_' \
                  f'fc{args.model_params.factor}_eb{args.model_params.embed}_dt{args.model_params.distil}_' \
                  f'{args.des}_{ii}'

        print(f'>>>>>>> Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)

        if args.gpu.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu.gpu_type == 'cuda':
            torch.cuda.empty_cache()

def run_experiments_with_tracking(configs: List[ExperimentConfig], executed_file: str):
    executed = load_executed_experiments(executed_file)
    total = len(configs)

    for idx, config in enumerate(configs):
        config_hash = compute_config_hash(config)

        if config_hash in executed:
            print(f"Skipping already executed experiment: {executed[config_hash]} ({idx + 1}/{total})")
            continue

        print(f"Running experiment {config.experiment_id} ({idx + 1}/{total})")

        # Place your experiment execution code here
        run_experiment(config)

        executed[config_hash] = config.experiment_id
        save_executed_experiments(executed_file, executed)
        print(f"Completed experiment {config.experiment_id}")

if __name__ == '__main__':

    exp_configs = load_exp_configs("ETT_exp_test.json")
    run_experiments_with_tracking(exp_configs, executed_file="executed_exps.json")
        
