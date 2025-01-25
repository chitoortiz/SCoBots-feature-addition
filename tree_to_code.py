import joblib
from sklearn.tree import _tree

def tree_to_code(tree, feature_names, output_file):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    with open(output_file, 'w') as f:
        f.write("def decision_tree_predict(features):\n")

        def recurse(node, depth):
            indent = "    " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                f.write(f"{indent}if features['{name}'] <= {threshold}:\n")
                recurse(tree_.children_left[node], depth + 1)
                f.write(f"{indent}else:  # if features['{name}'] > {threshold}\n")
                recurse(tree_.children_right[node], depth + 1)
            else:
                f.write(f"{indent}return {tree_.value[node].argmax()}\n")

        recurse(0, 1)

# Load the decision tree model
model_path = './resources/viper_extracts/extract_output/Pong_seed0_reward-env_oc-extraction/Tree-16.1_best.viper'
decision_tree = joblib.load(model_path)

# Get the number of features from the decision tree
n_features = decision_tree.n_features_in_

# Generate dummy feature names if not explicitly known
feature_names = [f"feature{i}" for i in range(n_features)]

# Generate Python code from the decision tree and write to a file
output_file = 'decision_tree_predict.py'
tree_to_code(decision_tree, feature_names, output_file)

print(f"Decision tree code has been written to {output_file}")