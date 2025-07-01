import subprocess
import logging
from config_utils import load_exp_configs, compute_config_hash, load_executed_experiments, save_executed_experiments

from log_utils import setup_file_logging

logger = setup_file_logging(log_file_path="tspindorama_launcher.log", level=logging.INFO)

CONFIG_FILE = "generated_exp_configs.json"
EXECUTED_FILE = "executed_exps.json"

def main():
    configs = load_exp_configs(CONFIG_FILE)
    executed = load_executed_experiments(EXECUTED_FILE)

    total = len(configs)
    for idx, config in enumerate(configs):
        config_hash = compute_config_hash(config)
        if config_hash in executed:
            logger.info(f"Skipping already executed experiment: {executed[config_hash]} ({idx + 1}/{total})")
            continue

        logger.info(f"Launching experiment {config.experiment_id} ({idx + 1}/{total})")
        try:
            subprocess.run(
                ['python', 'runbr.py', '--hash', config_hash],
                check=True
            )
            executed[config_hash] = config.experiment_id
            save_executed_experiments(EXECUTED_FILE, executed)
            logger.info(f"Finished experiment {config.experiment_id}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Experiment failed: {config.experiment_id}")
            logger.exception(e)

if __name__ == '__main__':
    main()
