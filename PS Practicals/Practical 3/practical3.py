import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data
np.random.seed(42)
data = {
    'A': np.random.normal(0, 1, 100),
    'B': np.random.normal(0, 2, 100),
    'C': np.random.normal(0, 1.5, 100)
}
df = pd.DataFrame(data)

# a. Find the data distributions using box and scatter plot.
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=df)
plt.title('Box Plot')

plt.subplot(1, 2, 2)
sns.scatterplot(x='A', y='B', data=df)
plt.title('Scatter Plot')
plt.tight_layout()
plt.show()

# b. Find the outliers using plot.
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, showfliers=True)
plt.title('Outliers Detection')
plt.show()

# c. Plot the histogram, bar chart, and pie chart on sample data.
plt.figure(figsize=(15, 5))

# Histogram
plt.subplot(1, 3, 1)
sns.histplot(df['A'], bins=20, kde=True)
plt.title('Histogram')

# Bar Chart
plt.subplot(1, 3, 2)
df.sum().plot(kind='bar')
plt.title('Bar Chart')

# Pie Chart (using absolute values)
plt.subplot(1, 3, 3)
df.abs().sum().plot(kind='pie', autopct='%1.1f%%')
plt.title('Pie Chart')

plt.tight_layout()
plt.show()
