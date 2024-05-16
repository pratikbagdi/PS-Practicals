import pandas as pd

#program for mtcars
mtcars = pd.read_csv('mtcars.csv')

print("Summary Statistics for mtcars:")
print(mtcars.describe())

print("\nGeneral Information for mtcars:")
print(mtcars.info())

print("\nQuartile Information for mtcars:")
numeric_columns = mtcars.select_dtypes(include=['number']).columns
print(mtcars[numeric_columns].quantile([0.25, 0.5, 0.75]))

#program for cars
cars = pd.read_csv('cars.csv')

print("Summary Statistics for cars:")
print(mtcars.describe())

print("\nGeneral Information for cars:")
print(mtcars.info())

print("\nQuartile Information for cars:")
numeric_columns = mtcars.select_dtypes(include=['number']).columns
print(mtcars[numeric_columns].quantile([0.25, 0.5, 0.75]))

#program for cars
iris = pd.read_csv('iris.csv')

print("Original Iris dataset:")
print(iris.head())

subset_condition = (iris['sepal_length'] > 5.0) & (iris['sepal_width'] > 3.0)
print("\nSubset of Iris Dataset:")
print(subset_condition)

aggregate_result = iris.groupby('species').agg({'petal_length': 'mean'})
print("\nAggregate result - Mean petal length for each species:")
print(aggregate_result)
