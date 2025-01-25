import joblib
from sklearn.tree import export_graphviz
import graphviz

# Load the decision tree model
model_path = './resources/viper_extracts/extract_output/Pong_seed0_reward-env_oc-extraction/Tree-16.1_best.viper'
decision_tree = joblib.load(model_path)

# Export the decision tree to a DOT format
dot_data = export_graphviz(decision_tree, out_file=None, 
                           feature_names=None,  
                           class_names=None,  
                           filled=True, rounded=True,  
                           special_characters=True)  

# Visualize the decision tree using graphviz
graph = graphviz.Source(dot_data)  
graph.render("decision_tree")  # This will create a file named 'decision_tree.pdf'

# To display the tree in a Jupyter Notebook, you can use:
# from IPython.display import display
# display(graph)