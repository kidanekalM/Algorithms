import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./Iris.csv')

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
                elif y_train.iloc[indx] == class1:
                    weights = update_weights(weights, 1, record.to_numpy()[1:], CLASS_1, pred_num)

    return weights

def get_pred(X_train, y_train, X_test, y_test,epoch):
    correct_predictions = 0
    total_predictions = len(X_test)
    y_pred = []

    Iris_setosa = "Iris-setosa"
    Iris_virginica ="Iris-virginica"
    Iris_versicolor = "Iris-versicolor"
    weights_history1 = []
    weights_history2 = []
    # Epoch = 5
    weights_history1.append( [0,0,0,0,0])
    weights1 = perceptron(X_train,y_train,[0,0,0,0,0],Iris_versicolor,Iris_virginica)
    for i in range(epoch):
        weights1 = perceptron(X_train,y_train,weights1,Iris_versicolor,Iris_virginica)
        weights_history1.append( weights1)
   
    weights_history2.append( [0,0,0,0,0])
    weights2 = perceptron(X_train,y_train,[0,0,0,0,0],Iris_setosa,Iris_versicolor)
    for i in range(epoch):
        weights2 = perceptron(X_train,y_train,weights2,Iris_setosa,Iris_versicolor)
        weights_history2.append( weights2)

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
    
    # plot_decision_boundary(X_train, y_train, weights1, weights2, 'Final Decision Boundaries')
    return weights1,weights2,weights_history1,weights_history2,correct_predictions,total_predictions,y_pred
def calculate_performance(correct_predictions,total_predictions,y_pred,y_test):
    
    y_test_array = y_test.to_numpy()

    accuracy = accuracy_score(y_test_array, y_pred) 
    precision = precision_score(y_test_array, y_pred,average='macro') 
    recall = recall_score(y_test_array, y_pred,average='macro') 
    f1_score_value = f1_score(y_test_array, y_pred,average='macro') 

    return [accuracy, precision, recall, f1_score_value]



def plot_decision_boundary1(X, y, weights1, weights2, title):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    weights1 = np.array(weights1)
    weights2 = np.array(weights2)

    x_min, x_max = X.to_numpy()[:, 0].min() - 1, X.to_numpy()[:, 0].max() + 1
    y_min, y_max = X.to_numpy()[:, 1].min() - 1, X.to_numpy()[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z1 = (weights1[0] + weights1[1] * xx + weights1[2] * yy + weights1[3] + weights1[4])
    Z2 = (weights2[0] + weights2[1] * xx + weights2[2] * yy + weights2[3] + weights2[4])

    y_pred1 = np.where(Z1 >= 0, 1, 0)
    y_pred2 = np.where(Z2 >= 0, 1, 0)

    plt.figure(1, figsize=(10, 8))
    plt.contour(xx, yy, y_pred1, levels=[0], colors='blue', linewidths=2)
    plt.contour(xx, yy, y_pred2, levels=[0], colors='green', linewidths=2)

    plt.scatter(X.to_numpy()[:, 0], X.to_numpy()[:, 1], c=y_encoded, edgecolors='k', marker='o')

    x_vals = np.array([x_min, x_max])
    y_vals1 = -(weights1[0] + weights1[1] * x_vals) / weights1[2] + 50
    y_vals2 = -(weights2[0] + weights2[1] * x_vals) / weights2[2] + 45
    plt.plot(x_vals, y_vals1, color='blue', linestyle='--', linewidth=2, label='Weights 1')
    plt.plot(x_vals, y_vals2, color='green', linestyle='--', linewidth=2, label='Weights 2')

    plt.title(title)
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()
    plt.show()



def plot_convergence(weights_history1, weights_history2, epochs):

    epochs_range = list(range(epochs + 1))
    
    for i in range(len(weights_history1[0])):
        plt.figure(figsize=(12, 6))
        plt.plot(epochs_range, [weights[i] for weights in weights_history1], label=f'Weights1 Feature {i}')
        plt.plot(epochs_range, [weights[i] for weights in weights_history2], label=f'Weights2 Feature {i}')
        plt.xlabel('Epoch')
        plt.ylabel(f'Weight {i}')
        plt.title(f'Weight Convergence for Feature {i}')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_perceptron_convergence(error_history):

    fig, ax = plt.subplots(figsize=(8, 6))

    error_history = np.array(error_history)

    ax.plot(error_history, label='Error')
    ax.set_title('Perceptron Convergence')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.legend()

    plt.tight_layout()
    plt.show()



weights1,weights2,weights_history1,weights_history2,correct_predictions,total_predictions,y_pred = get_pred(X_train,y_train,X_test,y_test,5)
print(weights1,weights2,correct_predictions,total_predictions,y_pred)
a,p,r,f1 = calculate_performance(correct_predictions,total_predictions,y_pred,y_test)
print("Accuracy =",a*100,"%\nPrecision = ",p*100,"%\nRecall = ",r,"\nF1 score = ",f1)
plot_decision_boundary1(X_train,y_train,weights1, weights2,"Decision Boundary")

error_history = []
for i in range(8):
    weights1,weights2,weights_history1,weights_history2,correct_predictions,total_predictions,y_pred = get_pred(X_train,y_train,X_test,y_test,i)
    error_history.append(total_predictions- correct_predictions)
plot_perceptron_convergence(error_history)
