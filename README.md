# Breast-cancer-prediction
# Training the model
import numpy as np
import pandas as pd
import sklearn.datasets                                   # no need to download separate datasets. sklearn itself have all the datasets
breast_cancer = sklearn.datasets.load_breast_cancer()     #for extracting datasets
x = breast_cancer.data                                    # data column has all the data for predicting the person condition
y = breast_cancer.target                                  # target column says that the person has melignan or benign
data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
data["class"] = breast_cancer.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1)               # i need only 10 percent of the whole data for teting
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,stratify = y)  # for getting equal distrubution of mean of y 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,stratify = y,random_state = 1)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x,y)
# Testing the data by importing the new input 
input_data = (17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)
input_data_asaray = np.asarray(input_data)                        #converting the input into the numpy array 
print(input_data)
print()

reshaped_input_data = input_data_asaray.reshape(1,-1)              # reshaping the array 
print(reshaped_input_data)
predicted = classifier.predict(reshaped_input_data)                 # predicting the data
print()
print(predicted)

if (predicted[0] == 0):
    print("The patient is in danger condition")
else:
    print("The patient is in safe condition")
