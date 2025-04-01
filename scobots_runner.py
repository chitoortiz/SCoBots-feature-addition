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

    base_cmd = f"-g {config['game']} -s {config['seed']}"

    if config.get("reward"):
        base_cmd += f" -r {config['reward']}"
    if config.get("prune"):
        base_cmd += f" -p {config['prune']}"
    if config.get("exclude_properties", False):
        base_cmd += " -x"
    if config.get("hud", False):
        base_cmd += " --hud"
    if config.get("hackatari", False):
        base_cmd += " --hackatari"
    if config.get("mods"):
        base_cmd += f" -mods {config['mods']}"

    if mode == "train":
        command = f"python train.py {base_cmd} -env {config['environments']}"
        if config.get("progress", False):
            command += " --progress"
        if config.get("rgb", False):
            command += " --rgb"
        run_command(command)

    elif mode == "eval":
        command = f"python eval.py {base_cmd} -t {config['times']}"
        if config.get("progress", False):
            command += " --progress"
        if config.get("rgb", False):
            command += " --rgb"
        if config.get("viper", False):
            command += " --viper"
        run_command(command)

    elif mode == "render":
        command = f"python render_agent.py {base_cmd}"
        if config.get("record", False):
            command += " --record"
        if config.get("nb_frames", 0) > 0:
            command += f" --nb_frames {config['nb_frames']}"
        if config.get("print_reward", False):
            command += " --print-reward"
        if config.get("rgb", False):
            command += " --rgb"
        if config.get("viper", False):
            command += " --viper"
        run_command(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="runner_configurations.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    main(args.config)
