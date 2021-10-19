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
print('Figure 1. Diagram Heatmap')
# plt.show()

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
print('Figure 2. Diagram Bar')

print('\n##########\n')

# figure dibawah ini membandingkan antara volatile acidity terhadap Quality
plt.figure(figsize=(10,10))
sns.barplot(x='quality',y='volatile acidity',data=wine_data)
# plt.show()
print('Figure 3. Diagram Barplot')

print('\n##########\n')

# figure dibawah ini membandingkan antara ph terhadap Quality
plt.figure(figsize=(10,10))
sns.barplot(x='quality',y='pH',data=wine_data)
# plt.show()
print('Figure 4. Diagram Barplot')

print('\n##########\n')

# figure dibawah ini menghitung jumlah wine berdasarkan kualitas menggunakan count plot
plt.figure(figsize=(10,10))
sns.countplot(x="quality", data=wine_data)
print('Figure 5. Diagram Countplot')
plt.show()
# Beritag komentar pada plt.show() diatas untuk menyembunyikan hasil diagram dan langsung melakukan Splitting Data

print('\n##########\n')

# # Splitting Data

# memisahkan kolom quality pada tabel dan memberikan label
# 1 = good wine
# 0 = bad wine
X = wine_data.drop(['quality'],axis=1)
y = wine_data['quality'].apply(lambda quality: 1 if quality >= 6 else 0)

# memisahkan data menjadi training data dan testig data
# 80% dari data akan digunakan sebagai training data dan 20% akan digunakan sebagai testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 2)
print(X.shape,X_train.shape)

print('\n##########\n')

# # Model Building

# penggunaan random forest model
model = RandomForestClassifier() # deklarasi random forest
model.fit(X_train,y_train)
print(model)

print('\n##########\n')

# # Prediction and Evaluation of the Model

# Menggunakan model untuk melakukan prediksi dan melakukan pengecekan tingkat akurasi
train_pred = model.predict(X_train)
print(train_pred)
Training_score = accuracy_score(train_pred,y_train) # melakukan perbandingan dengan y_train yang asli dan hasil prediksi lalu menghitung perbedaan/error
print("Accuracy Score (Data Training):",Training_score) # output adalah hasil tingkat akurasi pada penggunaan data training
print()
test_pred = model.predict(X_test)
print(test_pred)
Test_score = accuracy_score(test_pred,y_test) 
print("Accuracy Score (Data Testing):",Test_score)

# # Kesimpulan
# 1. Workflow pengerjaan = 
#  # a. melakukan import terhadap library dan dataset yang akan digunakan
#  # b. melakukan pengecekan jika terdapat missing data
#  # c. melakukan Exploratry Data Analysis
#  # d. melakukan split data terhadap beberapa variabel
#  # d. melakukan improvisasi terhadap model
# 2. Pada umumnya, tingkat akurasi 75% sudah cukup baik, tetapi masih bisa diperbaiki hingga mendapat tingkat akurasi yang lebih baik
# 3. Hasil output dari Random Forest Clasification, dapat digunakan sebagai decision making, pada kasus ini, hasil output dapat digunakan untuk menentukan kombinasi yang akan digunakan dalam pembuatan Red Wine dengan kualitas terbaik