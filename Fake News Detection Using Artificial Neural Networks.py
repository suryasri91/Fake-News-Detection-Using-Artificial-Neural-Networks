import numpy as np
import pandas as pd
import matplotlib.pyplot as plt;

import os
os.getcwd()

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#library to clean the text
import re

#natural language tool kit library
import nlkt

#library used to remove stop words
nlkt.download('stopwards')

from nlkt.corpus import stopwords
#library used for streaming

from nlkt.stem.porter import PorterStemmer
import df

#replacing punctuations and numbers using re library
review = df['Review'][0]
review = re.sub('[^a-zA-Z']','',review)

#converting declared variable into lowercase
review = review.lower()

#library use3d to remove stop words
nlkt.download('stopwords')
from nlkt.corpus import stopwords
#create an object for stemming
ps = PorterStemmer()
#split the variable
review = review.split()
#apply stemming
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

#creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
#To extract max 1500 features
cv = CountVectorizer(max_features=1500)
#x contains vectorised data (in dependent variable)
x = cv.fit_transform(data).toarray()

#y as dependent variable
y = df.iloc[:,1].values

#splitting the data into x_train, x_test, y_train, y_test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

#import library which uses tensor flow as backend
import keras
#library to initialize the model
from keras.models import Sequential
#Library used to add layers
from keras.layers import Dense

model=sequential()

model.add(Dense(output_dim=1550, init='uniform', activation = 'relu', input_dim = 1500))

model.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))

model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = 8, epochs = 3)

model.save('mymodel.h5')# this will save the weights, for keras h5 is extension

#fit Linear Regression Model to the dataset
from sklearn import linear_model as lm

#Create the Linear Regression object
model = lm.LinearRegression()

#Fit the object to the dataset
alg = model.fit(x_train, y_train)

accuracy = model.score(x_train, y_train)
print("Accuracy of the mode : ", accuracy*100)

from sklearn.tree import DecisionTreeRegressor

#Fit Decision Tree Regression Model ton the dataset
from sklearn.tree import DecisionTreeRegressor as dtr

#Create the Decision tree regressor object
regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(x_train, y_train)

accuracy = regressor.score(x_train, y_train) 
print("Accuracy of the model : ", accuracy*100)