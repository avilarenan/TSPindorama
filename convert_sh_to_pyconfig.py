import shlex
import re
import json
import yaml
from typing import List
from exp_params import ExperimentConfig, DataConfig, ForecastConfig, ModelConfig, OptimizationConfig


def parse_value(value: str):
    """Attempt to parse value as int, float, or leave as string."""
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_shell_command(shell_command: str) -> ExperimentConfig:
    """
    Parses a single shell script Python call into an ExperimentConfig dataclass.
    """
    tokens = shlex.split(shell_command, posix=True)

    args = {}
    i = 0
    while i < len(tokens):
        if tokens[i].startswith('--'):
            key = tokens[i][2:]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith('--'):
                value = parse_value(tokens[i + 1])
                args[key] = value
                i += 2
            else:
                args[key] = True  # Handle flag arguments
                i += 1
        else:
            i += 1

    # Build the nested dataclass config
    config = ExperimentConfig(
        task_name=args.get('task_name', 'long_term_forecast'),
        is_training=int(args.get('is_training', 1)),
        model_id=args.get('model_id', 'test'),
        model=args.get('model', 'Autoformer'),
        des=args.get('des', 'test'),

        data=DataConfig(
            name=args.get('data', 'ETTh1'),
            root_path=args.get('root_path', './data/ETT/'),
            data_path=args.get('data_path', 'ETTh1.csv'),
            features=args.get('features', 'M')
        ),
        forecast=ForecastConfig(
            seq_len=int(args.get('seq_len', 96)),
            label_len=int(args.get('label_len', 48)),
            pred_len=int(args.get('pred_len', 96))
        ),
        model_params=ModelConfig(
            e_layers=int(args.get('e_layers', 2)),
            d_layers=int(args.get('d_layers', 1)),
            factor=int(args.get('factor', 1)),
            enc_in=int(args.get('enc_in', 7)),
            dec_in=int(args.get('dec_in', 7)),
            c_out=int(args.get('c_out', 7))
        ),
        optimization=OptimizationConfig(
            itr=int(args.get('itr', 1))
        )
    )

    return config


def extract_python_commands_from_shell(shell_script: str) -> List[str]:
    """
    Extract all python commands from a shell script, supporting multi-line commands
    with or without backslash continuations.
    """
    # This pattern captures each python block until the next python block or end of file.
    pattern = r'(python -u run\.py[\s\S]*?)(?=(\n\n|$|\npython -u run\.py))'
    matches = re.findall(pattern, shell_script)

    commands = []
    for match in matches:
        command_block = match[0]
        cleaned = ' '.join(command_block.strip().splitlines())
        commands.append(cleaned)

    return commands


def parse_shell_script_file(filepath: str) -> List[ExperimentConfig]:
    """
    Process an entire .sh file and return a list of ExperimentConfig objects.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    commands = extract_python_commands_from_shell(content)
    configs = [parse_shell_command(command) for command in commands]
    return configs


def save_configs_to_json(configs: List[ExperimentConfig], output_file: str):
    """
    Save a list of ExperimentConfig objects to a JSON file.
    """
    from dataclasses import asdict
    with open(output_file, 'w') as f:
        json.dump([asdict(config) for config in configs], f, indent=4)


def save_configs_to_yaml(configs: List[ExperimentConfig], output_file: str):
    """
    Save a list of ExperimentConfig objects to a YAML file.
    """
    from dataclasses import asdict
    with open(output_file, 'w') as f:
        yaml.dump([asdict(config) for config in configs], f, sort_keys=False)


# Example usage
if __name__ == '__main__':
    shell_script_path = './scripts/long_term_forecast/ETT_script/Autoformer_ETTh1.sh'
    configs = parse_shell_script_file(shell_script_path)

    save_configs_to_json(configs, 'Autoformer_ETTh1.json')
    save_configs_to_yaml(configs, 'Autoformer_ETTh1.yaml')

    for i, config in enumerate(configs):
        print(f'\nExperiment {i + 1}:')
        print(config)
