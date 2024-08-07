import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/PC/Desktop/New folder/slides/semester 10 - spring/cs488 AI/assing/Algorithms/Iris.csv')

X = df.drop('Species', axis=1)  
y = df['Species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
#X_train, X_test, y_train, y_test = next(StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42).split(X, y))

def manhattan_distance(rec1, rec2):
    dist = 0
    for i in range(1,len(rec1)):
        dist += abs(rec1[i] - rec2[i])
    return dist

def euclidean_distance(rec1, rec2):
    dist = 0
    for i in range(1,len(rec1)-1):
        dist += (rec1[i] - rec2[i])**2
    return dist ** 0.5
def knn(training_set,training_class, new_record,k,distance_metrics):
  
    ts_array= training_set.to_numpy()
    distances = [[0 for _ in range(2)] for _ in range(len(ts_array))]   
    for i in range(len(ts_array)):
        distances[i][0] = training_class.iloc[i]
        if(distance_metrics == "man"):
            distances[i][1] =manhattan_distance(ts_array[i],new_record)
        elif(distance_metrics == "euc"):
            distances[i][1] =euclidean_distance(ts_array[i],new_record)
    sorted_distances = sorted(distances, key=lambda x: x[1])[:k]

    Iris_setosa = 0
    Iris_virginica = 0
    Iris_versicolor = 0

    for val in sorted_distances:
        if(val[0] == "Iris-setosa"):
            Iris_setosa += 1
        elif (val[0] == "Iris-virginica"):
            Iris_virginica += 1
        elif (val[0] == "Iris-versicolor"):
            Iris_versicolor+=1
    if((Iris_setosa+Iris_virginica+Iris_versicolor)!= len(sorted_distances)):
        print(Iris_setosa+Iris_virginica+Iris_versicolor," ",len(sorted_distances))
        print("ERRORR it does not add up!")
        
    max_value = max(Iris_setosa,Iris_versicolor,Iris_virginica)
    if(max_value == Iris_setosa):
        return "Iris-setosa"
    elif(max_value == Iris_versicolor):
        return "Iris-versicolor"
    elif(max_value == Iris_virginica):
        return "Iris-virginica"
        
    
        
        
# print(knn(X_train,y_train,X_train.loc[49],5,"euc"))
# print(knn(X_train,y_train,X_train.iloc[X_train['Id'] == 50],5))
#print(X_train.loc[49]) to see the id by train

def calculate_accuracy(X_train, y_train, X_test, y_test, k,distance_metrics):
    
    correct_predictions = 0
    total_predictions = len(X_test)
    
    for i, test_val in enumerate(X_test.iterrows()):
        _, test_row = test_val
        predicted_label = knn(X_train, y_train, test_row, k,distance_metrics)
        if predicted_label == y_test.iloc[i]:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy

print(calculate_accuracy(X_train,y_train,X_test,y_test,1,"man"))
































import numpy as np

def knn(X_train, y_train, test_row, k, distance_metric):
    # Existing knn function implementation
    pass

def calculate_accuracy(X_train, y_train, X_test, y_test, k, distance_metric):
    correct_predictions = 0
    total_predictions = len(X_test)
    y_pred = []

    for i, test_row in enumerate(X_test.iterrows()):
        _, test_row = test_row
        predicted_label = knn(X_train, y_train, test_row, k, distance_metric)
        y_pred.append(predicted_label)
        if predicted_label == y_test.iloc[i]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    precision_score = precision(y_test, y_pred)
    recall_score = recall(y_test, y_pred)
    f1_score_value = f1_score(y_test, y_pred)

    return accuracy, precision_score, recall_score, f1_score_value

def precision(y_true, y_pred):
    """
    Calculates the precision of a classification model.
    """
    true_positives = 0
    false_positives = 0

    for i in range(len(y_true)):
        if y_pred[i] == y_true[i] and y_pred[i] == 1:
            true_positives += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            false_positives += 1

    if true_positives + false_positives == 0:
        return 0
    else:
        return true_positives / (true_positives + false_positives)

def recall(y_true, y_pred):
    """
    Calculates the recall of a classification model.
    """
    true_positives = 0
    false_negatives = 0

    for i in range(len(y_true)):
        if y_pred[i] == y_true[i] and y_pred[i] == 1:
            true_positives += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            false_negatives += 1

    if true_positives + false_negatives == 0:
        return 0
    else:
        return true_positives / (true_positives + false_negatives)

def f1_score(y_true, y_pred):
    """
    Calculates the F1 score of a classification model.
    """
    precision_score = precision(y_true, y_pred)
    recall_score = recall(y_true, y_pred)

    if precision_score + recall_score == 0:
        return 0
    else:
        return 2 * (precision_score * recall_score) / (precision_score + recall_score)