import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

df = pd.read_csv('C:/Users/PC/Desktop/New folder/slides/semester 10 - spring/cs488 AI/assing/Algorithms/Iris.csv')

X = df.drop('Species', axis=1)  
y = df['Species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
def activation_function(bias,inputs,weights,class0,class1):
    prediction = bias * weights[0]
    for i in range(len(weights)-1):
        
        prediction += inputs[i]*weights[i+1]
    
    if prediction > 0:
        return 1,class1
    else: 
        return 0,class0

def update_weights(weights,bias,inputs,exp,pred):
    weights[0] += 1*(exp-pred)*bias
    for i in range(len(weights)-1):
        weights[i+1] += 1*(exp-pred)*inputs[i]
    return weights

def perceptron(X_train,y_train,class0,class1):
    weights = [0,0,0,0,0]
    CLASS_0 = 0
    CLASS_1 = 1

    for indx,record in enumerate(X_train.iterrows()):
        _,record = record
        pred_num,pred_label = activation_function(1,record.to_numpy()[1:],weights,class0,class1)
        print(y_train.iloc[indx] , class0, pred_label, class0)
        if (y_train.iloc[indx] == class0) and (class0  != pred_label):
            weights = update_weights(weights,1,record.to_numpy()[1:],CLASS_0,pred_num)
        else:
            weights = update_weights(weights,1,record.to_numpy()[1:],CLASS_1,pred_num)
    return weights



weights1 = perceptron(X_train,y_train,"Iris-setosa","not")
weights2 = perceptron(X_train,y_train,"Iris-virginica","Iris-versicolor")
for indx,record in enumerate(X_test.iterrows()):
    _,record = record
    _,pred = activation_function(1,X_test.iloc[indx].to_numpy()[1:],weights1,"Iris-setosa","not")
    if pred == "not":
        _,pred = activation_function(1,X_test.iloc[indx].to_numpy()[1:],weights2,"Iris-virginica","Iris-versicolor")
        print(y_test.iloc[indx],pred)
    else:
        print(y_test.iloc[indx],pred)


