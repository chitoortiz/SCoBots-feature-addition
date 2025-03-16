import yaml
import subprocess
import argparse

def load_config(config_file):
    """Load YAML configuration file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def run_command(command):
    """Execute a shell command."""
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

def main(config_path):
    config = load_config(config_path)
    
    mode = config.get("mode", "train")
    
    if mode == "train":
        # Prepare arguments for the training mode
        command = f"python train.py -g {config['game']} -s {config['seed']} -env {config['env_num']} -r {config['reward']} --progress"
        run_command(command)
    
    elif mode == "eval":
        # Prepare arguments for the evaluation mode
        command = f"python eval.py -g {config['game']} -s {config['seed']} -t {config['times']} -r {config['reward']} -p {config['prune']}"
        run_command(command)
    
    elif mode == "render":
        # Prepare arguments for the render mode
        command = f"python render_agent.py -g {config['game']} -s {config['seed']} -r human -p {config['prune']} --viper"
        run_command(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configurations.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    
    main(args.config)
