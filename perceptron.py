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

def perceptron(X_train,y_train,weights,class0,class1):
    # weights = [0,0,0,0,0]
    CLASS_0 = 0
    CLASS_1 = 1

    # for indx,record in enumerate(X_train.iterrows()):
    #     _,record = record
    #     pred_num,pred_label = activation_function(1,record.to_numpy()[1:],weights,class0,class1)
    #     if (y_train.iloc[indx] == class0) and (class0  != pred_label):
    #         weights = update_weights(weights,1,record.to_numpy()[1:],CLASS_0,pred_num)
    #     elif (y_train.iloc[indx] == class1) and (class1  != pred_label):
    #         weights = update_weights(weights,1,record.to_numpy()[1:],CLASS_1,pred_num)
    #     else:
    #         print("weights")
    # return weights
    for indx, record in enumerate(X_train.iterrows()):
        _, record = record
        pred_num, pred_label = activation_function(1, record.to_numpy()[1:], weights, class0, class1)
        if (y_train.iloc[indx] == class0) or (y_train.iloc[indx] == class1):
            if y_train.iloc[indx] != pred_label:
            
                if y_train.iloc[indx] == class0:
                    weights = update_weights(weights, 1, record.to_numpy()[1:], CLASS_0, pred_num)
                else:#if y_train.iloc[indx] == class1:
                    weights = update_weights(weights, 1, record.to_numpy()[1:], CLASS_1, pred_num)

    return weights


def calculate_performance(X_train, y_train, X_test, y_test):
    correct_predictions = 0
    total_predictions = len(X_test)
    y_pred = []
    y_test_array = y_test.to_numpy()

    weights1 = perceptron(X_train,y_train,"Iris-setosa","not")
    weights2 = perceptron(X_train,y_train,"Iris-virginica","Iris-versicolor")
    
    for indx,record in enumerate(X_test.iterrows()):
        _,record = record
        _,pred = activation_function(1,X_test.iloc[indx].to_numpy()[1:],weights1,"Iris-setosa","not")
        if pred == "not":
            _,pred = activation_function(1,X_test.iloc[indx].to_numpy()[1:],weights2,"Iris-virginica","Iris-versicolor")
        # print(pred,y_test.iloc[indx],correct_predictions)
        if pred == y_test.iloc[indx]:
            correct_predictions += 1
        y_pred.append(pred)
        print(indx,pred,y_test.iloc[indx])


    # for i, test_row in enumerate(X_test.iterrows()):
    #     _, test_row = test_row
    #     predicted_label = knn(X_train, y_train, test_row, k, distance_metric)
    #     y_pred.append(predicted_label)
    #     if predicted_label == y_test.iloc[i]:
    #         correct_predictions += 1
    
    accuracy = accuracy_score(y_test_array, y_pred) #correct_predictions / total_predictions
    precision = precision_score(y_test_array, y_pred,average='macro') #precision(y_test_array, y_pred)
    recall = recall_score(y_test_array, y_pred,average='macro') #recall(y_test_array, y_pred)
    f1_score_value = f1_score(y_test_array, y_pred,average='macro') #f1_score(y_test_array, y_pred)

    return [accuracy, precision, recall, f1_score_value]

Iris_setosa = "Iris-setosa"
Iris_virginica ="Iris-virginica"
Iris_versicolor = "Iris-versicolor"

n = "not"
weights1 = perceptron(X_train,y_train,[0,0,0,0,0],Iris_versicolor,Iris_virginica)
weights1 = perceptron(X_train,y_train,weights1,Iris_versicolor,Iris_virginica)
weights1 = perceptron(X_train,y_train,weights1,Iris_versicolor,Iris_virginica)
weights1 = perceptron(X_train,y_train,weights1,Iris_versicolor,Iris_virginica)
weights1 = perceptron(X_train,y_train,weights1,Iris_versicolor,Iris_virginica)
weights1 = perceptron(X_train,y_train,weights1,Iris_versicolor,Iris_virginica)

weights2 = perceptron(X_train,y_train,[0,0,0,0,0],Iris_setosa,Iris_virginica)
weights3 = perceptron(X_test,y_train,weights1,Iris_versicolor,Iris_virginica)
print(weights1,weights2,weights3)
for indx,record in enumerate(X_test.iterrows()):
    _,record = record
    _,pred = activation_function(1,X_test.iloc[indx].to_numpy()[1:],weights1,Iris_versicolor,Iris_virginica)
    print(indx,y_test.iloc[indx],pred)
    # if pred == Iris_versicolor:
        # _,pred = activation_function(1,X_test.iloc[indx].to_numpy()[1:],weights2,Iris_setosa,Iris_virginica)
        # print(indx,y_test.iloc[indx],pred)
    # if pred == Iris_versicolor:
        # _,pred = activation_function(1,X_test.iloc[indx].to_numpy()[1:],weights3,Iris_versicolor,Iris_virginica)
        # print(indx,y_test.iloc[indx],pred)
    # else:
        # print(indx,y_test.iloc[indx],pred)

     

# print(calculate_performance(X_train,y_train,X_test,y_test))