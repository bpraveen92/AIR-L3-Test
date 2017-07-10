# -*- coding: utf-8 -*-
"""
Created on Mon Jul 03 16:36:50 2017

@author: Praveen
"""

"""Import necessary packages"""
import sys, traceback
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Fits random forest classifier on training data and returns test prediction results
def rf_clf_fit(X_train,y_train,X_test):
    rfc = RandomForestClassifier(n_estimators=300,max_features = 'auto',random_state = 42)
    rfc.fit(X_train,y_train)
    predictions = rfc.predict(X_test)
    return predictions

#Function that performs kfold cross validation on random forest classifier
def rf_kfold(iris_df):
    rfc = RandomForestClassifier(n_estimators=300,max_features = 'auto',random_state=42) #--> total no. of trees to be built before making a prediction
    cv_scores = cross_val_score(rfc,iris_df.drop('Labels',axis=1), iris_df['Labels'], cv=10)
    return np.mean(cv_scores)

#Fits KNN classifier on training data and returns test prediction results.
def knn_fit(X_train,y_train,X_test):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)
    pred = knn.predict(X_test)
    return pred
    
#Function runs knn classification models on various neighbor parameters and thus plots error rate variations.
def knn_new_neighbors(X_train,y_train,X_test,y_test):
    error_rate = []
    for i in [10,20,50,80]:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
        
    print 'Error Rate analysis for various neighbors: \n'
    plt.figure(figsize=(10,4))
    plt.plot([10,20,50,80],error_rate,color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()
    return

#GridSearchCV is used to best estimate parameters for the classifier that basically would end up in high accuracy predictions on unseen data.
def grid_search_estimation(features,classes):
    dt_classifier = DecisionTreeClassifier(random_state=45)

    tune_parameters = {"criterion": ["gini", "entropy"],
                  "min_samples_split": [2, 10, 20],
                  "max_depth": [None, 2, 5, 10],
                  "min_samples_leaf": [1, 5, 10],
                  "max_leaf_nodes": [None, 5, 10, 20],
                  }
    
    cross_validation = StratifiedKFold(classes, n_folds=10)
    
    grid_search = GridSearchCV(dt_classifier,
                               param_grid=tune_parameters,
                               cv=cross_validation)
    
    grid_search.fit(features, classes)
    print 'GridSearchCV estimation: \n'
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    return 

#KFold cross validation for the decision tree classifier model. Returns average accuracy score.
def k_fold_cv(iris_df):
    dt_classifier = DecisionTreeClassifier(random_state=45)
    cv_scores = cross_val_score(dt_classifier,iris_df.drop('Labels',axis=1), iris_df['Labels'], cv=10)
    return np.mean(cv_scores)

#Simplistic model performance measure across several iterations. (Works best only for small data sets)
def try_different_samples(features,classes):
    model_performance = []
    for repetition in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(features,classes,
                                                    test_size=0.30)
        dt_classifier = DecisionTreeClassifier()
        dt_classifier.fit(X_train, y_train)
        accuracy = dt_classifier.score(X_test, y_test)
        model_performance.append(accuracy)
    return model_performance

#Fits decision tree classifier on training data and returns classifier object.
def get_clf_fit(X_train,y_train):
    dt_classifier = DecisionTreeClassifier(random_state=45)
    return dt_classifier.fit(X_train, y_train)
   
#Splits training and test data.
def train_test(features,classes):
    X_train, X_test, y_train, y_test = train_test_split(features,classes,
                                                    test_size=0.30,random_state=45)
    return X_train, X_test, y_train, y_test


def main():
    
    filepath = str(raw_input('Enter Iris_data.csv filepath:'))
    #sample filepath = C:\Users\Praveen\Desktop\DataScience\L3 test\iris_data.csv
    
    try:
    
        """Load dataset into dataframe"""
        iris_df = pd.read_csv(filepath, na_values=['NA'])
        iris_df.head()
    
        """Looking at some basic descriptive stats"""
        iris_df.describe().to_string()
        """Drop the NULL columns"""
        iris_df.drop(iris_df.columns[[5,6]], axis=1, inplace=True)
        
        print "Exploratory analysis: \n"
        iris_df.loc[iris_df['Labels'] == 0, 'Sepal Width'].hist()
        plt.title('Checking for outliers in Sepal Width values')
        plt.show()
        iris_df.loc[iris_df['Labels'] == 2, 'Sepal Length'].hist()
        plt.title('Checking for outliers in Sepal Length values')
        plt.show()
        
        print 'Removing any data point less than 2.5 for Sepal Width - outliers.'
        print 'Removing any data point less than 5.5 for Sepal Length - outliers.'
        iris_df = iris_df.loc[(iris_df['Labels'] != 0) | (iris_df['Sepal Width'] >= 2.5)]
        iris_df = iris_df.loc[(iris_df['Labels'] != 2) | (iris_df['Sepal Length'] >= 5.5)]
        
        iris_df = iris_df.rename(columns=lambda x: x.strip())
        features = iris_df[['Sepal Length', 'Sepal Width','Petal Length', 'Petal Width']].values
        classes = iris_df['Labels'].values
        
        X_train, X_test, y_train, y_test = train_test(features,classes)
        decision_tree = get_clf_fit(X_train,y_train)
        print "Decision tree model prediction score: "+str(decision_tree.score(X_test, y_test))
        
        Acc = try_different_samples(features,classes)
        print "Average classification accuracy for decision tree model after 1000 iterations: "+str(np.mean(Acc))
        
        avg_kfold = k_fold_cv(iris_df)
        print "Average classification accuracy for decision tree model after 10 fold cross validation: "+str(avg_kfold)+'\n'
        
        grid_search_estimation(features,classes)
        pred = knn_fit(X_train,y_train,X_test)
        print "Confusion matrix for KNN model with n_neighbors=1 prediction: \n"
        print(confusion_matrix(y_test,pred))
        
        print "Classification report for KNN model with n_neighbors=1 prediction: \n"
        print(classification_report(y_test,pred))
        
        knn_new_neighbors(X_train,y_train,X_test,y_test)
        
        random_forest_pred = rf_clf_fit(X_train,y_train,X_test)
        print "Confusion matrix for random forest model: \n"
        print(confusion_matrix(y_test,random_forest_pred))
        print "Classification report for random forest model: \n"
        print(classification_report(y_test,random_forest_pred))
        
        avg_rf_cv = rf_kfold(iris_df)
        print "Average classification accuracy for random forest model after 10 fold cross validation: "+str(avg_rf_cv)+'\n'
    
    except Exception:
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        pymsg = "PYTHON ERRORS:\nTraceback info:\n" + tbinfo + "\nError Info:\n" + str(sys.exc_info()[1])
        d = datetime.datetime.now()
        log = open('ERROR_Log_Iris_Classification.txt',"w")
        log.write("\n")
        log.write("__________________________________________")
        log.write("ERROR LOGS ")
        log.write("__________________________________________")
        log.write("\n")
        log.write("Log: " + str(d) + "\n")
        log.write("" + pymsg + "\n")
        log.close()
        print 'Unexpected exit !!! Check log file (ERROR_Log.txt)'

if __name__ == '__main__':
    main()