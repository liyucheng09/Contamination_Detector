import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

evaluation_base = 'wiki'

with open(f'reports/perplexity_{evaluation_base}.json') as f:
    data_dict = json.load(f)

# Creating a DataFrame from JSON data
df = pd.DataFrame.from_dict(data_dict, orient='index')

metrics_data = {
    'memorised': ('s', 'navy'),
    'clean': ('+', 'navy'),
    'XSum': ('x', 'violet'),
}

fig, ax = plt.subplots(figsize=(8, 2.8), dpi=150)

# Create a horizontal scatter plot for each metric
for benchmark, numbers in df.iteritems():
    marker_style, color = metrics_data[benchmark]
    models, perplexities = list(numbers.index), list(numbers.values)
    plt.scatter(perplexities, models, label=benchmark, s=20, marker=marker_style, color=color)

# Adjust plot
plt.ylabel('Models', fontweight='bold')
plt.xlabel('Perplexity', fontweight='bold')
plt.legend( loc='upper right', bbox_to_anchor=(1.05, 1.0), ncol=1, fontsize=8)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
