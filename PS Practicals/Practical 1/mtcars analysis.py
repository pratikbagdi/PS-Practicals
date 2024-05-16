import pandas as pd

mtcars = pd.read_csv('mtcars.csv')

print("Summary Statistics for mtcars:")
print(mtcars.describe())

print("\nGeneral Information for mtcars:")
print(mtcars.info())

print("\nQuartile Information for mtcars:")
numeric_columns = mtcars.select_dtypes(include=['number']).columns
print(mtcars[numeric_columns].quantile([0.25, 0.5, 0.75]))

