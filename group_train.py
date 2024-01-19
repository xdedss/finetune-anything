
import os
import yaml
import sys
import jinja2
import itertools
import prettytable
import json
import threading
import time
import subprocess


def expand_config_template(config_workspace: str):

    def find_exp_name(config_str: str):
        config_obj = yaml.safe_load(config_str)
        return config_obj['train']['experiment_name']

    params_config_file = os.path.join(config_workspace, 'params.yaml')
    with open(params_config_file, 'r', encoding='utf-8') as f:
        params_config = yaml.safe_load(f)

    template_file = os.path.join(config_workspace, 'template.yaml')
    with open(template_file, 'r', encoding='utf-8') as f:
        template_str = f.read()

    print(json.dumps(params_config, indent=4, ensure_ascii=False))

    keys = [k for k in params_config]
    combos = list(itertools.product(*[params_config[k] for k in params_config]))

    # pretty print
    print(f'{len(combos)} combinations.')
    table = prettytable.PrettyTable(('i', *keys))
    for i, combo in enumerate(combos):
        table.add_row((i, *combo))
    print(table)

    run_commands = []
    for combo in combos:
        template_copy = template_str
        for key, value in zip(keys, combo):
            template_copy = template_copy.replace(f"${key}", str(value))
        # find exp name
        exp_name = find_exp_name(template_copy)
        exp_yaml_path = os.path.join(config_workspace, f'{exp_name}.yaml')
        exp_log_path = os.path.join(config_workspace, f'{exp_name}.out.log')
        with open(exp_yaml_path, 'w', encoding='utf-8') as f:
            f.write(template_copy)
        run_commands.append(f'python -u train.py --cfg "{exp_yaml_path}" > {exp_log_path} 2>&1')

    return run_commands

def run_multiple_commands_on_gpu(command_list):

    # Create a lock to control access to the commands list
    commands = [item for item in command_list]
    commands_lock = threading.Lock()

    def run_command(worker_id, cmd):
        try:
            print(f"Worker {worker_id} STARTING command: {cmd}")
            # Run the command as a shell command
            process = subprocess.Popen(f'CUDA_VISIBLE_DEVICES={worker_id} {cmd}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print(f"Worker {worker_id} successfully executed command: {cmd}")
            else:
                print(f"Worker {worker_id} encountered an error while running command: {cmd}")
                print(f"Error output:\n{stderr.decode()}")
        
        except Exception as e:
            print(f"Worker {worker_id} encountered an exception while running command: {cmd}")
            print(f"Exception details: {str(e)}")
    # Worker function
    def worker(worker_id):
        while True:
            # Acquire the lock to access the commands list
            with commands_lock:
                if not commands:
                    # No more commands to process
                    break
                # Get the next command
                cmd = commands.pop(0)
            
            # Run the command
            run_command(worker_id, cmd)

    # Create 4 worker threads
    num_workers = 4
    threads = []
    for i in range(num_workers):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all worker threads to finish
    for thread in threads:
        thread.join()

config_workspace = 'config/0109_batch_lr_and_schedule'
command_list = expand_config_template(config_workspace)
run_multiple_commands_on_gpu(command_list)



