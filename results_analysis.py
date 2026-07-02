import numpy as np
from pathlib import Path

# Specify the directory path
results_dir = Path("results")

data = np.loadtxt(results_dir / 'concatenated_results.csv', delimiter=',', fmt="%.2f")

# Analyze the concatenated data