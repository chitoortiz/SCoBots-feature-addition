import os
import glob
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Folder where the "Tree-*_best.viper" files are located
base_folder_path = 'viper_extracts/extract_output'

# Get all the .viper files recursively within the base folder
file_pattern = os.path.join(base_folder_path, '**', 'Tree-*best.viper')
matching_files = glob.glob(file_pattern, recursive=True)

# Iterate through the matching .viper files
for viper_path in matching_files:
    # Extract the folder name and generate the Python script name
    folder_name = os.path.basename(os.path.dirname(viper_path))
    script_name = f"{folder_name}-tree-rules.py"
    script_path = os.path.join(os.path.dirname(viper_path), script_name)
    
    # Check if the folder contains any .viper files
    if not os.path.exists(viper_path):
        print(f"No Tree-*_best.viper files found in folder {os.path.dirname(viper_path)}, skipping folder.")
        continue  # Skip processing this folder if no .viper files exist
    
    # Load the decision tree from the .viper file
    dtree = joblib.load(viper_path)

    # Get the number of features from the decision tree
    n_features = dtree.n_features_in_
    features = [f"feature_{i}" for i in range(n_features)]
    thresholds = dtree.tree_.threshold

    # Recursive function to extract decision rules
    def print_tree_rules(tree, feature_names, node=0, depth=0):
        # Check if the node is a leaf node
        if tree.tree_.feature[node] == -2:  # -2 indicates leaf
            print(f"{'  ' * depth}Leaf node: class={tree.classes_[tree.tree_.value[node].argmax()]}")
            return
        
        feature_name = feature_names[tree.tree_.feature[node]]
        threshold = tree.tree_.threshold[node]

        print(f"{'  ' * depth}Node {node}: if {feature_name} <= {threshold:.2f}:")
        print_tree_rules(tree, feature_names, tree.tree_.children_left[node], depth + 1)
        print(f"{'  ' * depth}else:")
        print_tree_rules(tree, feature_names, tree.tree_.children_right[node], depth + 1)

    # Print the rules starting from the root node (node 0)
    print_tree_rules(dtree, features)

    def extract_tree_rules_as_code(tree, feature_names, class_names, node=0, depth=1):
        # Recursively extract tree rules as Python if statements and return as a string.
        
        # Base case: If we reach a leaf node
        if tree.tree_.feature[node] == -2:  # -2 indicates leaf node
            class_value = class_names[tree.tree_.value[node].argmax()]
            return f"{'    ' * depth}return '{class_value}'\n"
        
        # Otherwise, extract the current feature index and threshold
        feature_index = tree.tree_.feature[node]
        threshold = tree.tree_.threshold[node]
        
        # Generate the if-else condition for this node
        condition = f"input_features[{feature_index}] <= {threshold:.2f}"
        
        # Generate the left and right branches (recursive calls for children)
        left_branch = extract_tree_rules_as_code(tree, feature_names, class_names, 
                                                 tree.tree_.children_left[node], depth + 1)
        right_branch = extract_tree_rules_as_code(tree, feature_names, class_names, 
                                                  tree.tree_.children_right[node], depth + 1)
        
        # Return the generated Python if statement with proper indentation
        return f"{'    ' * depth}if {condition}:\n{left_branch}{'    ' * depth}else:\n{right_branch}"

    def generate_tree_code(dtree, feature_names, class_names):
        # Generate Python code from decision tree rules.
        
        code = "def predict(input_features):\n"
        code += extract_tree_rules_as_code(dtree, feature_names, class_names)
        return code

    # Generate Python code
    tree_code = generate_tree_code(dtree, features, dtree.classes_)

    # Save the generated code to the Python file in the correct directory
    with open(script_path, "w") as f:
        f.write(tree_code)

    print(f"Python code has been saved to {script_path}")

"""
# Assuming `dtree` is your trained DecisionTreeClassifier
plt.figure(figsize=(10, 8))
plot_tree(dtree, filled=True, 
          feature_names=None,
          class_names=None,
          rounded=True, fontsize=10, max_depth=7)
plt.show()
"""