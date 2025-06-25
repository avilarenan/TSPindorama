import shlex
import re
import json
import yaml
import os
from typing import List, Tuple, Dict
from exp_params import ExperimentConfig, DataConfig, ForecastConfig, ModelConfig, OptimizationConfig
from config_utils import save_configs_to_json, save_configs_to_yaml


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


MULTI_VALUE_ARGS = {'p_hidden_dims'}


def preprocess_shell_script(shell_script: str) -> str:
    """
    Preprocess the shell script to join lines split by trailing backslashes.
    """
    lines = shell_script.splitlines()
    joined_lines = []
    current_line = ''

    for line in lines:
        stripped = line.rstrip()
        if stripped.endswith('\\'):
            current_line += stripped[:-1] + ' '
        else:
            current_line += stripped
            joined_lines.append(current_line)
            current_line = ''

    if current_line:
        joined_lines.append(current_line)

    return '\n'.join(joined_lines)


def extract_variable_assignments(shell_script: str) -> Dict[str, str]:
    variable_pattern = r'^(\w+)=(["\']?[^"\']+?["\']?)$'
    array_pattern = r'^(\w+)\s*=\s*\((.*?)\)$'

    variables = {}
    arrays = {}

    for line in shell_script.splitlines():
        line = line.strip()
        array_match = re.match(array_pattern, line)
        if array_match:
            var_name = array_match.group(1)
            array_values = array_match.group(2).split()
            arrays[var_name] = array_values
            continue

        match = re.match(variable_pattern, line)
        if match:
            var_name = match.group(1)
            var_value = match.group(2).strip('"\'')
            variables[var_name] = var_value

    variables.update(arrays)
    return variables


def resolve_variables(command: str, variables: Dict[str, str]) -> str:
    for var, value in variables.items():
        if isinstance(value, list):
            continue
        command = re.sub(rf'\${{{var}}}', value, command)
        command = re.sub(rf'\${var}\b', value, command)
    return command


def expand_loops(shell_script: str, variables: Dict[str, str]) -> List[str]:
    expanded_commands = []

    indexed_loop_pattern = r'for\s+(\w+)\s+in\s+"\${!(\w+)\[@\]}";?\s*do([\s\S]*?)done'
    indexed_loops = re.findall(indexed_loop_pattern, shell_script, re.DOTALL)

    for loop_var, array_var, loop_body in indexed_loops:
        array_values = variables.get(array_var, [])
        for i, _ in enumerate(array_values):
            temp_body = loop_body.replace(f'${{{loop_var}}}', str(i))
            temp_body = re.sub(rf'\${array_var}\[\$\{{{loop_var}\}}\]', array_values[i], temp_body)
            temp_body = ' '.join(temp_body.strip().splitlines())
            expanded_commands.append(temp_body)

    direct_loop_pattern = r'for\s+(\w+)\s+in\s+"\${(\w+)\[@\]}";?\s*do([\s\S]*?)done'
    direct_loops = re.findall(direct_loop_pattern, shell_script, re.DOTALL)

    for loop_var, array_var, loop_body in direct_loops:
        array_values = variables.get(array_var, [])
        for val in array_values:
            temp_body = loop_body.replace(f'${{{loop_var}}}', val)
            temp_body = ' '.join(temp_body.strip().splitlines())
            expanded_commands.append(temp_body)

    return expanded_commands


def parse_shell_command(shell_command: str) -> ExperimentConfig:
    tokens = shlex.split(shell_command, posix=True)

    args = {}
    i = 0
    while i < len(tokens):
        if tokens[i].startswith('--'):
            key = tokens[i][2:]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith('--'):
                if key in MULTI_VALUE_ARGS:
                    values = []
                    i += 1
                    while i < len(tokens) and not tokens[i].startswith('--'):
                        values.append(parse_value(tokens[i]))
                        i += 1
                    args[key] = values
                else:
                    value = parse_value(tokens[i + 1])
                    args[key] = value
                    i += 2
            else:
                args[key] = True
                i += 1
        else:
            i += 1

    try:
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
    except Exception as e:
        raise ValueError(f"Error parsing command: {shell_command}\nReason: {e}")


def extract_python_commands_from_shell(shell_script: str) -> List[str]:
    pattern = r'(python -u run\.py[\s\S]*?)(?=(\n\n|$|\npython -u run\.py))'
    matches = re.findall(pattern, shell_script)

    commands = []
    for match in matches:
        command_block = match[0]
        cleaned = ' '.join(command_block.strip().splitlines())
        commands.append(cleaned)

    return commands


def parse_shell_script_file(filepath: str) -> Tuple[List[ExperimentConfig], List[str]]:
    with open(filepath, 'r') as f:
        raw_content = f.read()

    content = preprocess_shell_script(raw_content)
    variables = extract_variable_assignments(content)
    commands = extract_python_commands_from_shell(content)
    loop_expansions = expand_loops(content, variables)

    all_commands = commands + loop_expansions

    configs = []
    errors = []

    for command in all_commands:
        try:
            resolved_command = resolve_variables(command, variables)
            config = parse_shell_command(resolved_command)
            configs.append(config)
        except Exception as e:
            errors.append(str(e))

    return configs, errors


def parse_all_shell_scripts_in_folder(folder_path: str) -> Tuple[List[ExperimentConfig], List[str]]:
    all_configs = []
    all_errors = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.sh'):
            file_path = os.path.join(folder_path, filename)
            configs, errors = parse_shell_script_file(file_path)
            all_configs.extend(configs)
            all_errors.extend([f"File: {filename}\n{error}" for error in errors])

    return all_configs, all_errors


if __name__ == '__main__':
    shell_scripts_folder = './scripts/long_term_forecast/ETT_script'  # Change this to your folder path
    configs, errors = parse_all_shell_scripts_in_folder(shell_scripts_folder)

    save_configs_to_json(configs, 'ETT_exp_configs.json')
    save_configs_to_yaml(configs, 'ETT_exp_configs.yaml')

    print(f"\nSuccessfully parsed {len(configs)} experiments.")
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors:
            print(f"\n{error}")
