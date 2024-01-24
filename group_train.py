
import itertools
import json
import os
import subprocess
import sys
import threading
import time

import fire
import jinja2
import prettytable
import yaml
import file_lock

from rich.live import Live
from rich.table import Table

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
        exp_yaml_path = os.path.join(config_workspace, f'gen_{exp_name}.yaml')
        exp_log_path = os.path.join(config_workspace, f'gen_{exp_name}.out.log')
        with open(exp_yaml_path, 'w', encoding='utf-8') as f:
            f.write(template_copy)
        run_commands.append(f'python -u train.py --cfg "{exp_yaml_path}" > {exp_log_path} 2>&1')

    return run_commands, keys, combos

def run_multiple_commands_on_gpu(command_list, num_workers: int, on_status_change=None):

    def on_status_change_wrap(row: int, s: str):
        if (on_status_change is not None):
            on_status_change(row, s)

    # Create a lock to control access to the commands list
    commands = [(i, item) for i, item in enumerate(command_list)]
    commands_lock = threading.Lock()

    def run_command(worker_id, cmd_tuple):
        cmd_id, cmd = cmd_tuple
        try:
            print(f"Worker {worker_id} STARTING command: {cmd}")
            on_status_change_wrap(cmd_id, f'RUNNING[{worker_id}]')
            # Run the command as a shell command
            process = subprocess.Popen(f'CUDA_VISIBLE_DEVICES={worker_id} {cmd}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print(f"Worker {worker_id} successfully executed command: {cmd}")
                on_status_change_wrap(cmd_id, 'SUCCESS')
            else:
                print(f"Worker {worker_id} encountered an error while running command: {cmd}")
                print(f"Error output:\n{stderr.decode()}")
                on_status_change_wrap(cmd_id, f'ERROR {process.returncode}')
        
        except Exception as e:
            print(f"Worker {worker_id} encountered an exception while running command: {cmd}")
            print(f"Exception details: {str(e)}")
            on_status_change_wrap(cmd_id, f'ERROR')
    # Worker function
    def worker(worker_id):
        while True:
            # Acquire the lock to access the commands list
            with commands_lock:
                if not commands:
                    # No more commands to process
                    break
                # Get the next command
                cmd_tuple = commands.pop(0)
            
            # Run the command
            run_command(worker_id, cmd_tuple)

    threads = []
    for i in range(num_workers):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all worker threads to finish
    for thread in threads:
        thread.join()

def main(config_workspace):
    command_list, keys, combos = expand_config_template(config_workspace)
    status = ['PENDING' for _ in command_list]

    # rich table
    def generate_table() -> Table:
        """Make a new table."""
        table = Table()
        table.add_column("ID")
        for k in keys:
            table.add_column(k)
        table.add_column("Status")

        for row in range(len(command_list)):
            table.add_row(
                f"{row}", 
                *[str(item) for item in combos[row]],
                f'{status[row]}',
            )
        return table

    with Live(generate_table(), refresh_per_second=4) as live:

        def on_status_change(row: int, s: str):
            if ('error' in s.lower()):
                s = '[red]' + s
            if ('success' in s.lower()):
                s = '[green]' + s
            if ('running' in s.lower()):
                s = '[cyan]' + s
            status[row] = s
            live.update(generate_table())
        
        with file_lock.FileLock('group_train.lock'):
            run_multiple_commands_on_gpu(
                command_list, num_workers=4, on_status_change=on_status_change)

if __name__ == '__main__':
    fire.Fire(main)


