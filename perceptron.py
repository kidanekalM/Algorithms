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
    CLASS_0 = 0
    CLASS_1 = 1
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

def get_pred(X_train, y_train, X_test, y_test):
    correct_predictions = 0
    total_predictions = len(X_test)
    y_pred = []

    Iris_setosa = "Iris-setosa"
    Iris_virginica ="Iris-virginica"
    Iris_versicolor = "Iris-versicolor"
    Epoch = 5
    weights1 = perceptron(X_train,y_train,[0,0,0,0,0],Iris_versicolor,Iris_virginica)
    for i in range(Epoch):
        weights1 = perceptron(X_train,y_train,weights1,Iris_versicolor,Iris_virginica)

    weights2 = perceptron(X_train,y_train,[0,0,0,0,0],Iris_setosa,Iris_versicolor)
    for indx,record in enumerate(X_test.iterrows()):
        _,record = record
        _,pred = activation_function(1,X_test.iloc[indx].to_numpy()[1:],weights1,Iris_versicolor,Iris_virginica)
        # print(pred,y_test.iloc[indx],indx)
        if pred == Iris_versicolor:
            _,pred = activation_function(1,X_test.iloc[indx].to_numpy()[1:],weights2,Iris_setosa,Iris_versicolor)
        if pred == y_test.iloc[indx]:
            # print(pred,y_test.iloc[indx],indx)
            correct_predictions += 1
        y_pred.append(pred)
    return weights1,weights2,correct_predictions,total_predictions,y_pred
def calculate_performance(correct_predictions,total_predictions,y_pred,y_test):
    
    y_test_array = y_test.to_numpy()

    accuracy = accuracy_score(y_test_array, y_pred) 
    precision = precision_score(y_test_array, y_pred,average='macro') 
    recall = recall_score(y_test_array, y_pred,average='macro') 
    f1_score_value = f1_score(y_test_array, y_pred,average='macro') 

    return [accuracy, precision, recall, f1_score_value]

def plot_decision_boundary(weights1, weights2,X_train):
    x = X_train.to_numpy()

    # Calculate y values for each set of weights
    # y1 = relu(np.dot(weights1, np.ones(len(weights1))))
    # y2 = relu(np.dot(weights2, np.ones(len(weights2))))
    y1 = weights1[0]*1 + weights1[1]*x[0] + weights1[2]*x[1] + weights1[3]*x[2] + weights1[4]*x[3] 
    y2 = weights2[0]*1 + weights2[1]*x[0] + weights2[2]*x[1] + weights2[3]*x[2] + weights2[4]*x[3] 
    # Plot the results
    plt.plot(x, y1 * np.ones_like(x), label='Weights1')
    plt.plot(x, y2 * np.ones_like(x), label='Weights2')
    plt.xlabel('Input')
    plt.ylabel('Activation')
    plt.title('Activation Function with Given Weights')
    plt.legend()
    plt.show()

weights1,weights2,correct_predictions,total_predictions,y_pred = get_pred(X_train,y_train,X_test,y_test)
print(weights1,weights2,correct_predictions,total_predictions,y_pred)
a,p,r,f1 = calculate_performance(correct_predictions,total_predictions,y_pred,y_test)
print("Accuracy =",a,"\nPrecision = ",p,"\nRecall = ",r,"\nF1 score = ",f1)
# plot_decision_boundary([weights1,weights2],X_test.to_numpy(),y_test.to_numpy(),y_pred)
plot_decision_boundary(weights1, weights2,X_train)