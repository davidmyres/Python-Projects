# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:52:03 2022

@author: dthomas
"""
import csv
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

# Creating the confusion matrix for each 
def confusionMatrix(predictions, classColumn):
    # Each matrix is written as [a, b, c, d]
    cm = [0, 0, 0, 0]
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(len(predictions)):    
        # True Positives
        if predictions[i] == 1 and classColumn[i] == 1:
            a += 1
        # False Negatives
        elif predictions[i] != 1 and classColumn[i] == 1:
            b += 1
        # False Positives
        elif predictions[i] == 1 and classColumn[i] != 1:
            c += 1
        # True Negatives
        else:
            d += 1
    cm[0] = a
    cm[1] = b
    cm[2] = c
    cm[3] = d
    return cm


def results(predictions, classColumn):
    cm = confusionMatrix(predictions, classColumn)
    # Accuracy 
    accuracy = (cm[0]+cm[3])/(cm[0] + cm[1] + cm[2] + cm[3])
    print("Accuracy:", accuracy)
    
    # Precision
    precision = cm[0]/(cm[0] + cm[2])
    print("Precision:", precision)
    
    # Recall
    recall = cm[0]/(cm[0] + cm[1])
    print("Recall:", recall)

def main():
    
    # Reading from file
    df = pd.read_csv("FraudDetectionTrainingSet.csv")
    # Creating a NB Classifier with this dataset
    FraudDetectionTrainingSet = np.array(df)
    numColumns = len(df.columns)
    # Creating a training set using all instances
    attColumns = np.array(FraudDetectionTrainingSet[:len(df)-3000, 1:numColumns - 1])
    classColumn = np.array(FraudDetectionTrainingSet[:len(df)-3000, numColumns - 1])
    validationSet = np.array(FraudDetectionTrainingSet[len(df)-3000:, 1:numColumns - 1])
    validationSetClass = np.array(FraudDetectionTrainingSet[len(df)-3000:, numColumns - 1])
    knn = KNeighborsClassifier(n_neighbors=100)
    knn.fit(attColumns, classColumn)
    nb = GaussianNB()
    nb.fit(attColumns, classColumn)
    decision = tree.DecisionTreeClassifier()
    decision = decision.fit(attColumns, classColumn)
    tree.plot_tree(decision)
    trainPredictKNN = knn.predict(validationSet)
    trainPredictNB = nb.predict(validationSet)
    trainPredictDT = decision.predict(validationSet)
    with open('FraudDetectionTestingSetSolutionSample.csv','w', newline='') as f:    
        writer = csv.writer(f)
        writer.writerow(['ID', 'Class', 'KNNTarget', 'NBTarget', 'DTTarget'])
        for i in range(3000):
            writer.writerow([5000 + i + 1, FraudDetectionTrainingSet[len(df)-3000+i][numColumns-1],trainPredictKNN[i], trainPredictNB[i], trainPredictDT[i]])
    
    print("K-Nearest Neighbors:")
    results(trainPredictKNN, validationSetClass)
    print()
    
    
    print("Naive Bayes:")
    results(trainPredictNB, validationSetClass)
    print()
    
    print("Decision Tree:")
    results(trainPredictDT, validationSetClass)
    print()

    testing = pd.read_csv('FraudDetectionTestingSetAttributes (2).csv')
    FraudDetectingTestSet = np.array(testing)
    testingSet = np.array(FraudDetectingTestSet[:len(testing), 1:numColumns-1])
    predictionsKNN = knn.predict(testingSet)
    predictionsNB = nb.predict(testingSet)
    predictionsDT = decision.predict(testingSet)
    with open('predictions.csv','w', newline='') as f:    
        writer = csv.writer(f)
        writer.writerow(['ID' , 'KNNTarget'])
        for i in range(len(predictionsNB)):
            writer.writerow([8000 + i + 1, predictionsKNN[i], predictionsNB, predictionsDT])
   
main()