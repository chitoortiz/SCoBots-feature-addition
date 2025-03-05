import os
import glob
import re
import joblib

base_folder_path = 'resources/viper_extracts/extract_output'
file_pattern = os.path.join(base_folder_path, '**', 'Tree-*best.viper')
matching_files = glob.glob(file_pattern, recursive=True)

# Extract the if-else from tree
def extract_tree_body_as_code(tree, feature_names, class_names, node=0, depth=1):
    # If leaf node
    if tree.tree_.feature[node] == -2:
        class_label_str = class_names[tree.tree_.value[node].argmax()]
        class_label_int = int(class_label_str)
        return f"{'    ' * depth}return {class_label_int}\n"

    feature_index = tree.tree_.feature[node]
    threshold = tree.tree_.threshold[node]
    condition = f"input_features[{feature_index}] <= {threshold:.2f}"

    left_code = extract_tree_body_as_code(
        tree, feature_names, class_names,
        tree.tree_.children_left[node],
        depth + 1
    )
    right_code = extract_tree_body_as_code(
        tree, feature_names, class_names,
        tree.tree_.children_right[node],
        depth + 1
    )

    code = (
        f"{'    ' * depth}if {condition}:\n"
        f"{'    ' * (depth+1)}self.decision_path.append(\"{condition} => True\")\n"
        f"{left_code}"
        f"{'    ' * depth}else:\n"
        f"{'    ' * (depth+1)}self.decision_path.append(\"{condition} => False\")\n"
        f"{right_code}"
    )
    return code

# Build script for each tree
def build_final_script(env_name, vecnorm_path, focus_file_path, seed, tree_body_code):
    raw_script = f'''\
from utils.renderer import Renderer
from scobi import Environment
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

seed = {seed}
vecnorm_path = "{vecnorm_path}"
ff_file_path = "{focus_file_path}"

env = Environment("{env_name}", focus_dir="", focus_file=ff_file_path,
                hide_properties=False,
                draw_features=True,
                reward=0)

_, _ = env.reset(seed=seed)
dummy_vecenv = DummyVecEnv([lambda: env])
env = VecNormalize.load(vecnorm_path, dummy_vecenv)
env.training = False
env.norm_reward = False

class Python_Model:
    def __init__(self):
        self.decision_path = []

    def predict(self, input_features, deterministic=True):
        self.decision_path = []
        return [int(self.if_checks(input_features))], None

    def if_checks(self, input_features):
        input_features = input_features[0]

{tree_body_code.rstrip()}

model = Python_Model()

renderer = Renderer(env, model, ff_file_path, record=False, nb_frames=None)
renderer.print_reward = False
renderer.run_code_version()
        '''
    import textwrap
    final_script = textwrap.dedent(raw_script)
    return final_script

# Main loop over files
for viper_path in matching_files:
    folder_dir = os.path.dirname(viper_path)
    folder_name = os.path.basename(folder_dir)
    folder_name_no_extraction = folder_name.replace("-extraction", "")

    target_dir = os.path.join("resources", "checkpoints", folder_name_no_extraction)
    os.makedirs(target_dir, exist_ok=True)

    script_name = f"{folder_name_no_extraction}-tree-rules.py"
    script_path = os.path.join(target_dir, script_name)

    if not os.path.exists(viper_path):
        print(f"No Tree-*_best.viper files found in {folder_dir}, skipping.")
        continue

    # get the environment name, focus file, and seed from the folder_name
    words = folder_name_no_extraction.split("_")
    game_name = words[0]
    env_name = f"ALE/{game_name}-v5"
    focus_file_path = f"pruned_{game_name.lower()}.yaml"

    # get seed
    if len(words) > 1:
        seed_str = re.sub(r"[^0-9]", "", words[1])  # strip non-digits
        if seed_str:
            seed_val = int(seed_str)
        else:
            seed_val = 0
    else:
        seed_val = 0

    vecnorm_path = "best_vecnormalize.pkl"

    # Load the DecisionTree and extract code
    dtree = joblib.load(viper_path)

    n_features = dtree.n_features_in_
    features = [f"feature_{i}" for i in range(n_features)]

    tree_body_code = extract_tree_body_as_code(
        dtree,
        features,
        dtree.classes_,
        node=0,
        depth=2
    )


    # Build the script and write
    full_script = build_final_script(
        env_name=env_name,
        vecnorm_path=vecnorm_path,
        focus_file_path=focus_file_path,
        seed=seed_val,
        tree_body_code=tree_body_code
    )

    with open(script_path, "w") as f:
        f.write(full_script)

    print(f"Generated Python script at: {script_path}")