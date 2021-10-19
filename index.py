# Library yang akan digunakan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine_data = pd.read_csv('data_wine/winequality-red.csv') #mengambil file csv pada folder data_wine
wine_data.head()
print('Data kualitas wine: ')
print(wine_data)

print('\n##########\n')

# melakukan cek terhadap missing value
print('Informasi mengenai data wine yang digunakan: \n') # cek info mengenai data yang akan digunakan
wine_data.info()
# wine_data.info()
print('\nPengecekan terhadap nilai null\n')
print(wine_data.isnull().sum())

print('\n##########\n')

# # Exploratory Data Analysis
corr = wine_data.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr,cbar=True,square=True,fmt='.1f',annot=True,cmap='Reds')
# plt.show()
# hilangkan tag komentar pada plt.show untuk menampilkan diagram

print('\n##########\n')

qs = wine_data['quality'].unique() # melihat data pada kolom kualitas
print(sorted(qs))

print('\n##########\n')

# Membandingkan beberapa kandungan dengan kualitas
# figure dibawah ini membandingkan antara Sulphates terhadap Quality
plt.figure(figsize=(10,10))
x = wine_data.quality
y = wine_data.sulphates
plt.bar(x,y)
plt.xlabel("Quality")
plt.ylabel("Sulphates")
# plt.show()

print('\n##########\n')

# figure dibawah ini membandingkan antara volatile acidity terhadap Quality
plt.figure(figsize=(10,10))
sns.barplot(x='quality',y='volatile acidity',data=wine_data)
# plt.show()

print('\n##########\n')

# figure dibawah ini membandingkan antara ph terhadap Quality
plt.figure(figsize=(10,10))
sns.barplot(x='quality',y='pH',data=wine_data)
# plt.show()

print('\n##########\n')

# figure dibawah ini menghitung jumlah wine berdasarkan kualitas menggunakan count plot
plt.figure(figsize=(10,10))
sns.countplot(x="quality", data=wine_data)
# plt.show()
