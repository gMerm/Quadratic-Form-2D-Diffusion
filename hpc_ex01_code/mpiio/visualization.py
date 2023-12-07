import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


f = open("density_seq_vis.dat", "rb")
a = np.fromfile(f, dtype=np.float64)
df = pd.DataFrame(a)
sns.heatmap(df, cmap='viridis', annot=False, fmt="1.8f")

# Add labels and title
plt.xlabel('Double Values')
plt.ylabel('Density')
plt.title('Density Plot of Double Values')

# Show the plot
plt.show()
