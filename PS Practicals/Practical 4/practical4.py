import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import f_oneway

# Load the iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]

# a. Find the correlation matrix
correlation_matrix = iris_df.corr(numeric_only=True)
print("Correlation Matrix:")
print(correlation_matrix)

# b. Plot the correlation plot
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Plot of Iris Dataset')
plt.show()

# c. Analysis of covariance (ANOVA) for categorical variable 'species'
feature_columns = iris_df.columns[:-1]
for feature in feature_columns:
    anova_results = f_oneway(
        iris_df[feature][iris_df['species'] == 'setosa'],
        iris_df[feature][iris_df['species'] == 'versicolor'],
        iris_df[feature][iris_df['species'] == 'virginica']
    )
    print(f"\nANOVA results for '{feature}':")
    print("F-statistic:", anova_results.statistic)
    print("P-value:", anova_results.pvalue)

# Visualize boxplots for each feature based on species
plt.figure(figsize=(15, 8))
for i, feature in enumerate(feature_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='species', y=feature, data=iris_df)
    plt.title(f'{feature} vs Species')

plt.tight_layout()
plt.show()
