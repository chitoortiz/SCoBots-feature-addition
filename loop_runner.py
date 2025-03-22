import yaml
import subprocess
import itertools
import argparse

def load_config(config_file):
    """Load YAML configuration file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def run_command(command):
    """Execute a shell command."""
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

def generate_combinations(config):
    """Generate all combinations of parameters."""
    games = config.get("games", ["Pong"])
    seeds = config.get("seeds", [0])
    rewards = config.get("rewards", ["env"])
    prune = config.get("prune", [None, "default", "external"])  # Allow prune to be None
    rgb = config.get("rgb", [False])
    hud = config.get("hud", [False])

    # Generate all combinations of the parameters
    combinations = list(itertools.product(games, seeds, rewards, prune, rgb, hud))

    return combinations

def build_command(mode, game, seed, reward, prune, rgb, hud, config):
    """Build the command line for training, evaluation, or rendering."""
    command = f"-g {game} -s {seed} -r {reward}"
    
    # Add optional prune parameter
    if prune:
        command += f" -p {prune}"

    # Add other parameters
    if hud:
        command += " --hud"
    if rgb:
        command += " --rgb"
    if config.get("exclude_properties", False):
        command += " -x"
    if config.get("progress", False):
        command += " --progress"
    if config.get("hackatari", False):
        command += " --hackatari"
    if config.get("mods"):
        command += f" -mods {config['mods']}"
    
    # Mode-specific parameters
    if mode == "train":
        command += f" -env {config['environments']}"
    elif mode == "eval":
        command += f" -t {config.get('times', 10)}"
    elif mode == "render":
        if config.get("record", False):
            command += " --record"
        if config.get("nb_frames", 0):
            command += f" --nb_frames {config['nb_frames']}"
        if config.get("viper", False):
            command += " --viper"
        if config.get("print_reward", False):
            command += " --print-reward"

    return command

def train_with_combination(config, game, seed, reward, prune, rgb, hud):
    """Train with a specific combination of parameters."""
    print(f"\n=== Training {game} (Seed: {seed}, Reward: {reward}, Prune: {prune}, RGB: {rgb}, HUD: {hud}) ===\n")
    
    command = build_command("train", game, seed, reward, prune, rgb, hud, config)
    run_command(f"python train.py {command}")

def eval_with_combination(config, game, seed, reward, prune, rgb, hud):
    """Evaluate with a specific combination of parameters."""
    print(f"\n=== Evaluating {game} (Seed: {seed}, Reward: {reward}, Prune: {prune}, RGB: {rgb}, HUD: {hud}) ===\n")
    
    command = build_command("eval", game, seed, reward, prune, rgb, hud, config)
    run_command(f"python eval.py {command}")

def render_with_combination(config, game, seed, reward, prune, rgb, hud):
    """Render with a specific combination of parameters."""
    print(f"\n=== Rendering {game} (Seed: {seed}, Reward: {reward}, Prune: {prune}, RGB: {rgb}, HUD: {hud}) ===\n")
    
    command = build_command("render", game, seed, reward, prune, rgb, hud, config)
    run_command(f"python render_agent.py {command}")

def run_mode(config, mode):
    """Run training, evaluation, or render mode."""
    combinations = generate_combinations(config)
    
    for game, seed, reward, prune, rgb, hud in combinations:
        if mode == "train":
            train_with_combination(config, game, seed, reward, prune, rgb, hud)
        elif mode == "eval":
            eval_with_combination(config, game, seed, reward, prune, rgb, hud)
        elif mode == "render":
            render_with_combination(config, game, seed, reward, prune, rgb, hud)

def main(config_path):
    config = load_config(config_path)
    mode = config.get("mode", "train")
    run_mode(config, mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="loop_configurations.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    main(args.config)
