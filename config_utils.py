from typing import List, Tuple, Dict
from exp_params import ExperimentConfig, DataConfig, ForecastConfig, ModelConfig, OptimizationConfig
import yaml
import json
import datetime
import random
import os

NAME_LIST = [
    'alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel', 'india', 'juliet',
    'kilo', 'lima', 'mike', 'november', 'oscar', 'papa', 'quebec', 'romeo', 'sierra', 'tango',
    'uniform', 'victor', 'whiskey', 'xray', 'yankee', 'zulu',
    'aurora', 'breeze', 'cascade', 'donna', 'ember', 'falcon', 'glacier', 'harbor', 'island', 'jungle', 'kingdom',
    'legend', 'mirage', 'nebula', 'oasis', 'phoenix', 'quantum', 'raven', 'solstice', 'twilight', 'utopia',
    'vortex', 'wanderer', 'zenith', 'zephyr', 'onyx', 'crystal', 'ember', 'storm', 'echo', 'flare'
]

def generate_experiment_id() -> str:
    """
    Generate a unique experiment ID using the current date and two random names.
    """
    date_str = datetime.datetime.now().strftime('%Y%m%d')
    random_names = random.sample(NAME_LIST, 2)
    return f"{date_str}_{random_names[0]}_{random_names[1]}"

def load_configs_from_yaml(file_path: str) -> List[ExperimentConfig]:
    """
    Load experiment configurations from a YAML file and convert them to ExperimentConfig dataclass instances.
    """
    from dataclasses import asdict

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    configs = []
    for item in data:
        config = ExperimentConfig(
            task_name=item.get('task_name', 'long_term_forecast'),
            is_training=int(item.get('is_training', 1)),
            model_id=item.get('model_id', 'test'),
            model=item.get('model', 'Autoformer'),
            des=item.get('des', 'test'),
            data=DataConfig(**item.get('data', {})),
            forecast=ForecastConfig(**item.get('forecast', {})),
            model_params=ModelConfig(**item.get('model_params', {})),
            optimization=OptimizationConfig(**item.get('optimization', {}))
        )
        configs.append(config)

    return configs

def load_configs_from_json(file_path: str) -> List[ExperimentConfig]:
    """
    Load experiment configurations from a JSON file and convert them to ExperimentConfig dataclass instances.
    """
    from dataclasses import asdict

    with open(file_path, 'r') as f:
        data = json.load(f)

    configs = []
    for item in data:
        config = ExperimentConfig(
            task_name=item.get('task_name', 'long_term_forecast'),
            is_training=int(item.get('is_training', 1)),
            model_id=item.get('model_id', 'test'),
            model=item.get('model', 'Autoformer'),
            des=item.get('des', 'test'),
            data=DataConfig(**item.get('data', {})),
            forecast=ForecastConfig(**item.get('forecast', {})),
            model_params=ModelConfig(**item.get('model_params', {})),
            optimization=OptimizationConfig(**item.get('optimization', {}))
        )
        configs.append(config)

    return configs

def load_exp_configs(config_file_path):
    """ File can be either yaml of json"""

    exp_configs = []

    if config_file_path.endswith(".yaml"):
        loaded_yaml_configs = load_configs_from_yaml('ETT_exp_configs.yaml')
        for i, config in enumerate(loaded_yaml_configs):
            exp_configs += [config]
    elif config_file_path.endswith(".json"):
        loaded_json_configs = load_configs_from_json('ETT_exp_configs.json')
        for i, config in enumerate(loaded_json_configs):
            exp_configs += [config]
    else:
        raise Exception(f"Unexpected file type: {config_file_path.split('.')[1]}. Available types are: yaml or json. ")
    
    return exp_configs


def compute_config_hash(config: ExperimentConfig) -> str:
    """
    Compute a hash for the configuration excluding model_id and experiment_id.
    """
    import hashlib
    from dataclasses import asdict

    config_dict = asdict(config)
    config_dict.pop('model_id', None)
    config_dict.pop('experiment_id', None)
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()

def load_executed_experiments(file_path: str) -> Dict[str, str]:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def save_executed_experiments(file_path: str, executed: Dict[str, str]):
    with open(file_path, 'w') as f:
        json.dump(executed, f, indent=4)