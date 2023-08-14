'''
In this Code, we are performing Classifier Performance evaluation after applying Backward Search. The 2 classifiers we will be using for evaluation are Linear 
Discriminant Analysis(LDA) and Random Forest Classifier(RDC). We will import the data and perform preprocessing steps on it. We will then reduce the dimensions
of the data one by one by removing the dimension which results in the least error rate. On the reduced data we will perform classification using both LDA and RDC. 
During the classification process, for evaluation we will calculate the Training Time, Testing Time, Scores(Accuracy) and the Confusion Matrix for each Dimension 
reduced.
'''

#Importing Libraries
import time
import numpy
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Mean Centering and scaling the data to unit variance
def normalize(column):
    for i in column:
        X_dat[i] = X_dat[i]-X_dat[i].mean()
        X_dat[i] = X_dat[i]/X_dat[i].std()

#Initializing variables for Scores, Training Time, Testing Time and Confusion Matrix
features_to_remove = 18
lda_conf_matrices = {}
rf_conf_matrices = {}
lda_pc_accuracies = {}
rf_pc_accuracies = {}
lda_train_time = {}
lda_test_time = {}
rf_train_time = {}
rf_test_time = {}

def result_recordings(classification_type, cols_left, actual, prediction):
    """
        - Used to generate the confusion matrix
    :param classification_type: what type classifier is being used
    :param cols_left: Dimensions of the remaining dataset
    :param actual: Actual Label values
    :param prediction: Prediction vector
    """
    conf_matrix = numpy.zeros((2, 2), dtype=int)
    for elements in range(0, actual.shape[0]):
        if actual[elements] == 0 and prediction[elements] == 0:
            conf_matrix[1, 0] = conf_matrix[1, 0] + 1
        elif actual[elements] == 0 and prediction[elements] == 1:
            conf_matrix[0, 0] = conf_matrix[0, 0] + 1
        elif actual[elements] == 1 and prediction[elements] == 0:
            conf_matrix[1, 1] = conf_matrix[1, 1] + 1
        else:
            conf_matrix[0, 1] = conf_matrix[0, 1] + 1
    if classification_type == 'lda':
        lda_conf_matrices[cols_left] = conf_matrix
    else:
        rf_conf_matrices[cols_left] = conf_matrix

def best_accuracy_calculator_lda(x, y):
    """ Gives accuracy of prediction using Training and Test data respectively
        - Dividing all the data generated into training and testing sets
        - Float value of 0.25 signifies the proportion of test data
        - The value of random state is set so that results are the same each time, can be changed to None to get
        different sets of data on random seeds
        - Also takes record of the time being spent on each iterative fit and test
    :param x: Defines the independent variables/attributes
    :param y: Defines the dependent variable/attribute
    :return: the prediction accuracy score with given dependent and independent variable
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    lda = LDA()
    start_train_time = time.time()
    lda.fit(x_train, y_train)
    stop_train_time = time.time()
    train_time = stop_train_time - start_train_time

    start_test_time = time.time()
    prediction = lda.predict(x_test)
    stop_test_time = time.time()
    test_time = stop_test_time - start_test_time

    # Calculation of Accuracy
    error_num = sum(prediction != y_test)
    accuracy_pc = ((y_test.shape[0] - error_num) / y_test.shape[0]) * 100
    return accuracy_pc, train_time, test_time


def best_accuracy_calculator_rfc(x, y):
    """ Gives accuracy of prediction using Training and Test data respectively
        - Dividing all the data generated into training and testing sets
        - Float value of 0.25 signifies the proportion of test data
        - The value of random state is set so that results are the same each time, can be changed to None to get
        different sets of data on random seeds
        - Also takes record of the time being spent on each iterative fit and test
    :param x: Defines the independent variables/attributes
    :param y: Defines the dependent variable/attribute
    :return: the prediction accuracy score with given dependent and independent variable
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    rf_clf = RandomForestClassifier()
    start_train_time = time.time()
    rf_clf.fit(x_train, y_train)
    stop_train_time = time.time()
    train_time = stop_train_time - start_train_time

    start_test_time = time.time()
    prediction = rf_clf.predict(x_test)
    stop_test_time = time.time()
    test_time = stop_test_time - start_test_time

    # Calculation of Accuracy
    error_num = sum(prediction != y_test)
    accuracy_pc = ((y_test.shape[0] - error_num) / y_test.shape[0]) * 100
    return accuracy_pc, train_time, test_time

#Calculates Classification Time for Training and Testing for LDA
def lda_classifier_timer(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    lda = LDA()
    start_train = time.time()
    lda.fit(x_train, y_train)
    stop_train = time.time()
    lda_train_time[x.shape[1]] = stop_train - start_train

    start_test = time.time()
    prediction = lda.predict(x_test)
    stop_test = time.time()
    lda_test_time[x.shape[1]] = stop_test - start_test

    return prediction, y_test

#Calculates Classification Time for Training and Testing for RFC
def random_forest_classifier_timer(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    rf_clf = RandomForestClassifier()
    start_train = time.time()
    rf_clf.fit(x_train, y_train)
    stop_train = time.time()
    rf_train_time[x.shape[1]] = stop_train - start_train

    start_test = time.time()
    prediction = rf_clf.predict(x_test)
    stop_test = time.time()
    rf_test_time[x.shape[1]] = stop_test - start_test
    return prediction, y_test

def percentage_accuracy_calculator(y_test, prediction):
    """
        - Calculates accuracy percentage for the prediction
    :param y_test: actual labels
    :param prediction: Predicted labels
    :return: accuracy percentage
    """
    error_num = sum(prediction != y_test)
    accuracy_pc = ((y_test.shape[0] - error_num) / y_test.shape[0]) * 100
    return accuracy_pc

def selection(accuracy_scores, cols_remaining, classifier):
    """
        - Selects which dimension should be removed
        - Based upon removal of which dimension results in highest accuracy
    :param accuracy_scores: accuracy scores upon removing each column iteratively
    :param cols_remaining: Remaining dimensions
    :param classifier: Classifier being used
    :return:
    """
    accuracy_pc_values = accuracy_scores.values()
    max_pc_accuracy = max(accuracy_pc_values)
    # Column upon whose removal we get highest accuracy score is removed since that is least important
    col_index_to_remove = max(accuracy_scores, key=accuracy_scores.get)
    if classifier == 'lda':
        lda_pc_accuracies[cols_remaining] = max_pc_accuracy
    else:
        rf_pc_accuracies[cols_remaining] = max_pc_accuracy
    return col_index_to_remove

#Importing and Formating the data
X_dat = pd.read_csv("messidor_features.csv",names=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20'])
label = numpy.array(X_dat['x20']) #Assigning last column as the class values
X_dat.drop(columns=['x20'], inplace=True) #Dropping the class values from the data set
normalize(['x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18']) #Function call to mean-center and scale the data


X_lda = numpy.array(X_dat)
X_rf = numpy.array(X_dat)
for iterator in range(features_to_remove + 1):

    if iterator == 0 and X_lda.shape[1] == 19:
        prediction_lda, y_test_lda = lda_classifier_timer(X_lda, label)
        result_recordings('lda', X_lda.shape[1], y_test_lda, prediction_lda)
        accuracy = percentage_accuracy_calculator(y_test_lda, prediction_lda)
        lda_pc_accuracies[X_lda.shape[1]] = accuracy

    elif iterator > 0:
        temp_accuracy_scores = {}
        temp_iteration_times = [0, 0]
        for innerItr in range(X_lda.shape[1]):
            X_mod = numpy.delete(X_lda, innerItr, axis=1)
            accuracy, lda_temp_train_time, lda_temp_test_time = best_accuracy_calculator_lda(X_mod, label)
            temp_iteration_times[0] = temp_iteration_times[0] + lda_temp_train_time
            temp_iteration_times[1] = temp_iteration_times[1] + lda_temp_test_time
            temp_accuracy_scores[innerItr] = accuracy
        col_to_remove = selection(temp_accuracy_scores, X_lda.shape[1] - 1, 'lda')
        X_lda = numpy.delete(X_lda, col_to_remove, axis=1)
        prediction_lda, y_test_lda = lda_classifier_timer(X_lda, label)
        lda_train_time[X_lda.shape[1]] = temp_iteration_times[0]
        lda_test_time[X_lda.shape[1]] = temp_iteration_times[1]
        result_recordings('lda', X_lda.shape[1], y_test_lda, prediction_lda)
        accuracy = percentage_accuracy_calculator(y_test_lda, prediction_lda)
        temp_iteration_times = [0, 0]

for indexer in range(features_to_remove + 1):
    if indexer == 0 and X_rf.shape[1] == 19:
        prediction_rf, y_test_rf = random_forest_classifier_timer(X_rf, label)
        result_recordings('rfc', X_rf.shape[1], y_test_rf, prediction_rf)
        accuracy = percentage_accuracy_calculator(y_test_rf, prediction_rf)
        rf_pc_accuracies[X_rf.shape[1]] = accuracy

    elif indexer > 0:
        temp_accuracy_scores = {}
        temp_iteration_times = [0, 0]
        for innerItr in range(X_rf.shape[1]):
            X_mod = numpy.delete(X_rf, innerItr, axis=1)
            accuracy, rf_temp_train_time, rf_temp_test_time = best_accuracy_calculator_rfc(X_mod, label)
            temp_iteration_times[0] = temp_iteration_times[0] + rf_temp_train_time
            temp_iteration_times[1] = temp_iteration_times[1] + rf_temp_test_time
            temp_accuracy_scores[innerItr] = accuracy
        col_to_remove = selection(temp_accuracy_scores, X_rf.shape[1] - 1, 'rfc')
        X_rf = numpy.delete(X_rf, col_to_remove, axis=1)
        prediction_rf, y_test_rf = random_forest_classifier_timer(X_rf, label)
        rf_train_time[X_rf.shape[1]] = temp_iteration_times[0]
        rf_test_time[X_rf.shape[1]] = temp_iteration_times[1]
        result_recordings('rfc', X_rf.shape[1], y_test_rf, prediction_rf)
        accuracy = percentage_accuracy_calculator(y_test_rf, prediction_rf)
        temp_iteration_times = [0, 0]


print('LDA CONF MATRICES')
print(lda_conf_matrices)
print('LDA TRAIN TIME')
print(lda_train_time)
print('LDA TEST TIME')
print(lda_test_time)
print('LDA ACCURACIES')
print(lda_pc_accuracies)
print('RF CONFUSION MATRICES')
print(rf_conf_matrices)
print('RF TRAIN TIME')
print(rf_train_time)
print('RF TEST TIME')
print(rf_test_time)
print('RF ACCURACIES')
print(rf_pc_accuracies)
