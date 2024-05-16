import pandas as pd
import requests

txt_path='perimeter.txt'
csv_path='cars.csv'
txt_url='https://www.w3.org/TR/2003/REC-PNG-20031110/iso_8859-1.txt'
csv_url='https://gist.githubusercontent.com/bobbyhadz/9061dd50a9c0d9628592b156326251ff/raw/381229ffc3a72c04066397c948cf386e10c98bee/employees.csv'
excel_file_path = r'Dataforprac.xlsx'

# Opening TXT file from Disk
perimeter=open(txt_path,'r')
print("Txt File from Disk:  \n\n")
print(perimeter.read())
print('\n')

# Opening CSV file from Disk
cars=pd.read_csv(csv_path)
print("CSV File from Disk:  \n\n")
print(cars.head())
print('\n')

# # Opening TXT file from Web
response = requests.get(txt_url)
peri = response.text
print("Txt File from Web:  \n\n")
print(peri)
print('\n')

# Opening CSV file from Web
data = pd.read_csv(csv_url,sep=',',encoding='utf-8',)
print("CSV File from Web:  \n\n")
print(data.head())
print('\n')

df = pd.read_excel(excel_file_path)
print(df)
