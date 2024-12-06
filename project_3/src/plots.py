import matplotlib.pyplot as plt
import numpy as np

import json

# Path to the grid_search.json file
file_path = "../results/activation_search.json"

# Load the JSON file
with open(file_path, "r") as f:
    grid_search_results = json.load(f)

best_result = min(grid_search_results, key=lambda x: x['final_loss'])
print(best_result)