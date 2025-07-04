import argparse
import logging
from log_utils import setup_file_logging
from config_utils import load_exp_configs, compute_config_hash
from exp_params import ExperimentConfig
import torch
import torch.backends
import random
import numpy as np
from exp.exp_long_term_forecasting_br import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification

logger = setup_file_logging(log_file_path="tspindorama_single.log", level=logging.INFO)


def run_experiment(args):
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Setup device
    if torch.cuda.is_available() and args.gpu.use_gpu:
        device = torch.device(f'cuda:{args.gpu.gpu}')
        logger.info('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            device = torch.device("cpu")
        logger.info('Using cpu or mps')

    args.device = device

    if args.gpu.use_gpu and args.gpu.use_multi_gpu:
        device_ids = [int(id_) for id_ in args.gpu.devices.replace(' ', '').split(',')]
        args.device_ids = device_ids
        args.gpu.gpu = device_ids[0]

    logger.info('Args in experiment:')
    logger.info(args)

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

            logger.info(f'>>>>>>> Start training: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            logger.info(f'>>>>>>> Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)

            if args.gpu.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu.gpu_type == 'cuda':
                torch.cuda.empty_cache()
            del exp

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

        logger.info(f'>>>>>>> Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)

        if args.gpu.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu.gpu_type == 'cuda':
            torch.cuda.empty_cache()
        del exp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hash', required=True, help='Hash of the config to run')
    parser.add_argument('--config_file', default="generated_exp_configs.json", help='Config file to search in')
    args = parser.parse_args()

    configs = load_exp_configs(args.config_file)
    found = None
    for config in configs:
        if compute_config_hash(config) == args.hash:
            found = config
            break

    if not found:
        raise ValueError(f"No experiment config found for hash {args.hash}")

    logger.info(f"Running config with hash {args.hash}: {found.experiment_id}")
    run_experiment(found)

if __name__ == '__main__':
    main()
