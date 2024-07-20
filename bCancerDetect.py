# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 21:12:59 2024

@author: Abir & Karim
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from sklearn.preprocessing import StandardScaler

#PART 1
#read in data
breast_data = pd.read_csv("breastCancer.csv")

#PART 2
def clean_and_prep():
    # Drops empty cell rows
    good_data = breast_data.dropna()
    # Drops the ID column
    training_data = good_data.drop('id', axis=1)
    # Split the data
    training_set, testing_set = train_test_split(training_data, test_size = 0.2, random_state = 42)
    
    # Saves the training and testing diagnosis column
    global training_diagnosis
    training_diagnosis = training_set['diagnosis']
    global testing_diagnosis
    testing_diagnosis = testing_set['diagnosis']
    
    # Drops the diagnosis columns
    training_set = training_set.drop('diagnosis', axis=1)
    testing_set = testing_set.drop('diagnosis', axis=1)

    # Returns the cleaned up sets
    return training_set, testing_set

training_data, testing_data = clean_and_prep()

#PART 3
def train_tree():    
    # Training the tree
    time_start = time.time()
    dt = DecisionTreeClassifier().fit(training_data, training_diagnosis)
    time_total = time.time() - time_start
    
    # Drawing the decision tree
    plt.figure(figsize=(40, 20))
    plot_tree(dt, filled=True, feature_names=training_data.columns, class_names=['B', 'M'])
    plt.title('Breast Cancer Decision Tree')
    plt.show()
    
    # Evaluating the model
    dt_pred = dt.predict(testing_data)
    accuracy = accuracy_score(testing_diagnosis, dt_pred)
    cm = confusion_matrix(testing_diagnosis, dt_pred)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    
    # Model Performance
    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    
    # Print speed
    print(f"Training time: {time_total}")
    
    # Visualize the confusion matrix
    print(f"Confusion Matrix:\n{cm}")
    print("\n")

    
# train_tree()

#PART 4
def train_tree_with_depth(k):
    dt = DecisionTreeClassifier(max_depth=k).fit(training_data, training_diagnosis)
    dt_pred = dt.predict(testing_data)
    
    # Draw the decision tree
    plt.figure(figsize=(40,20))
    plot_tree(dt, filled=True, feature_names=training_data.columns, class_names=['B', 'M'])
    plt.title(f'Decision Tree (max_depth={k})')
    plt.show()
    
    # Evaluate the ensemble model
    accuracy = accuracy_score(testing_diagnosis, dt_pred)
    cm = confusion_matrix(testing_diagnosis, dt_pred)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    
    results = {
        'tree': dt,
        'predictions': dt_pred,
        'depth': k,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm
    }
    
    return results

def RFMRE():
    k_values = [3, 5, 7]
    
    time_start = time.time()
    with mp.Pool(processes=3) as pool:
        predictions = pool.map(train_tree_with_depth, k_values)
    time_total = time.time() - time_start
    
    print(f"Training time using MP: {time_total}: seconds")

    for result in predictions:
        plt.figure(figsize=(20, 10))
        plot_tree(result['tree'], filled=True, feature_names=training_data.columns, class_names=['B', 'M'])
        plt.title(f'Decision Tree (max_depth={result["depth"]})')
        plt.show()
        
        print(f"Decision Tree Results for max_depth={result['depth']}:")
        print(f"Accuracy: {result['accuracy']}")
        print(f"Sensitivity: {result['sensitivity']}")
        print(f"Specificity: {result['specificity']}")
        print(f"Confusion Matrix: \n{result['confusion_matrix']} \n")
        
    # Combine the arrays into a 2D array
    pred = [result['predictions'] for result in predictions]
    combined_arrays = np.array(pred)  
    majority_values = []
    
    # Loop through each index and find the majority value
    for i in range(combined_arrays.shape[1]):
        b_count = 0
        m_count = 0
        
        for array in combined_arrays:
            if array[i] == 'B':
                b_count += 1
            elif array[i] == 'M':
                m_count += 1
                
        if b_count > m_count:
            majority_values.append('B')
        else:
            majority_values.append('M')
    
    print("Majority Vote: ", majority_values)

    # Train and evaluate model on majority values array and testing set
    dt_majority = DecisionTreeClassifier().fit(testing_data, majority_values)
    dt_pred_majority = dt_majority.predict(testing_data)
    
    accuracy = accuracy_score(testing_diagnosis, dt_pred_majority)
    cm = confusion_matrix(testing_diagnosis, dt_pred_majority)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    
    print("\nRandom Forest Map-Reduce Ensemble Results for majority vote:")
    print(f"Accuracy: {accuracy}:")
    print(f"Sensitivity: {sensitivity}:")
    print(f"Specificity: {specificity}:")

    print(f"Confusion Matrix: \n{cm}")
    
# if __name__ == '__main__':   
#     RFMRE()

# PART 5
def train_svc():
    # Standardizing the features
    scaler = StandardScaler()
    training_data_scaled = scaler.fit_transform(training_data)
    testing_data_scaled = scaler.transform(testing_data)
    
    X = training_data_scaled[:, :2]
    Y = testing_data_scaled[:, :2]
    
    # Training the model
    start_time = time.time()
    svm = SVC(kernel='rbf', random_state=42).fit(X, training_diagnosis)
    time_total = time.time() - start_time
    
    # Evaluating the model
    svm_pred = svm.predict(Y)
    accuracy = accuracy_score(testing_diagnosis, svm_pred)
    cm = confusion_matrix(testing_diagnosis, svm_pred)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    
    # Model Performance
    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    
    # Print Speed
    print(f"Training Time: {time_total}")
    
    # Visualize the confusion matrix
    print(f"Confusion Matrix:\n{cm}")
    print("\n")
    
    # Visualize class boundary using 2 features
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                          np.arange(y_min, y_max, 0.01))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z[Z=='M'] = 1
    Z[Z=='B'] = 0
    Z = np.array(Z, dtype=float)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1])
    plt.title('SVM Classifier')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# train_svc()

#part 6
def make_plot(svm_class, scaler, kernel):
    training_data_scaled = scaler.transform(training_data)
    
    X = training_data_scaled[:, :2]
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    
    Z = svm_class.predict(xy)
    Z[Z=='M'] = 1
    Z[Z=='B'] = 0
    Z = np.array(Z, dtype=float)        
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1])
    plt.title(f'SVM {kernel} Classifier')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    
def evaluate(svm, scaler):
    testing_data_scaled = scaler.transform(testing_data)
    
    # Use only the first two features
    X_test = testing_data_scaled[:, :2]
    
    svm_pred = svm.predict(X_test)
    accuracy = accuracy_score(testing_diagnosis, svm_pred)
    cm = confusion_matrix(testing_diagnosis, svm_pred)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    
    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Confusion Matrix:\n{cm}")
    print("\n")

def train_svm(kernel):
    scaler = StandardScaler()
    training_data_scaled = scaler.fit_transform(training_data)
    
    # Use only the first two features
    X_train = training_data_scaled[:, :2]
    
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train, training_diagnosis)
    
    return svm, scaler

def triple_SVM():
    start_time = time.time()
    with mp.Pool(processes=3) as pool:
        results = pool.map(train_svm, ['rbf', 'linear', 'poly'])
    time_total = time.time() - start_time
    print(f"Training time using MP: {time_total} seconds")

    for (svm, scaler), kernel in zip(results, ['rbf', 'linear', 'poly']):
        print(f"\nResults for SVM with {kernel} kernel:")
        make_plot(svm, scaler, kernel)
        evaluate(svm, scaler)
    
    # Ensemble prediction using majority voting
    predictions = [svm.predict(scaler.transform(testing_data)[:, :2]) for svm, scaler in results]
    
    ensemble_pred = []
    for i in range(len(predictions[0])):
        votes = [pred[i] for pred in predictions]
        ensemble_pred.append(max(set(votes), key=votes.count))
        
    # print(predictions)
    # print(ensemble_pred)
    
      
    # Evaluate ensemble
    print("\nEnsemble Results:")
    accuracy = accuracy_score(testing_diagnosis, ensemble_pred)
    cm = confusion_matrix(testing_diagnosis, ensemble_pred)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    
    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Confusion Matrix:\n{cm}")

if __name__ == '__main__':
    triple_SVM()
















