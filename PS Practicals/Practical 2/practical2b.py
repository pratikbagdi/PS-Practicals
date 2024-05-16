import pandas as pd

# Specify the path to your Excel file without extra double quotes
excel_file_path = r'C:\Users\prati\OneDrive\Desktop\Dataforprac.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(excel_file_path)

# Display the DataFrame
print(df)
