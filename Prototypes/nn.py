import os
import urllib

import numpy as np
import csv
  
# Data sets
training = 'C:/Users/Indum/Documents/Year3/Programming/project/csv_files/normals_pngcsv_file'
rootdir = 'C:/Users/Indum/Documents/Year3/Programming/project/csv_files/'

#new csv with only the columns we want
f = open(rootdir + 'train', 'w')
header = ['count2', 'count3', 'count4', 'count5', 'count6','density']
writer1 = csv.writer(f)
writer1.writerow(header)
f.close()

# Load CSV and get only the columns we want and rewrite it to a new csv
raw_training = open(training, 'rt')
readertr = csv.reader(raw_training, delimiter=',')
x = 0
for row in readertr:
    x = x + 1
    if row != [] and x != 1:
        toadd = [row[2],row[3],row[4],row[5],row[6],row[7]]
        f = open(rootdir + 'train', 'a')
        writer = csv.writer(f)
        writer.writerow(toadd)
        f.close()
        
  
#tutorial starts here
        
import pandas as pd
df = pd.read_csv(rootdir + 'train')
datasettrain = df.values
#print(datasettrain)

#x values are input so first 5 columns
xtrain = datasettrain[:,:5]
#y values are output so last column
ytrain = datasettrain[:,5]

#print(xtrain)
#print(ytrain)

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(xtrain)

from sklearn.model_selection import train_test_split
#splits data into test, train and validation, 70 is train and 15 and 15 for other 2
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, ytrain, test_size=0.2)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)


from keras.models import Sequential
from keras.layers import Dense
model = Sequential([
    Dense(6, activation='relu', input_shape=(5,)),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))

print(model.evaluate(X_test, Y_test)[1])
