import pandas as pd

iris = pd.read_csv('iris.csv')

print("Original Iris dataset:")
print(iris.head())

subset_condition = (iris['sepal_length'] > 5.0) & (iris['sepal_width'] > 3.0)
print("\nSubset of Iris Dataset:")
print(subset_condition)

aggregate_result = iris.groupby('species').agg({'petal_length': 'mean'})
print("\nAggregate result - Mean petal length for each species:")
print(aggregate_result)
