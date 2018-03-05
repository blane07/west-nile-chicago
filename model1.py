# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 11:07:36 2018

@author: moisessalazar77
"""

from sklearn.cross_validation import cross_val_score

def print_accuracy_report(classifier, X, y, num_validations=5):
    print('Baseline:'+ str(round(100*baseline, 2)) + "%")
    accuracy = cross_val_score(classifier, 
            X, y, scoring='accuracy', cv=num_validations)
    print("Accuracy: " + str(round(100*accuracy.mean(), 2)) + "%")

    f1 = cross_val_score(classifier, 
            X, y, scoring='f1_weighted', cv=num_validations)
    print("F1: " + str(round(100*f1.mean(), 2)) + "%")

    precision = cross_val_score(classifier, 
            X, y, scoring='precision_weighted', cv=num_validations)
    print("Precision: " + str(round(100*precision.mean(), 2)) + "%")

    recall = cross_val_score(classifier, 
            X, y, scoring='recall_weighted', cv=num_validations)
    print("Specificity: " + str(round(100*recall.mean(), 2)) + "%")
