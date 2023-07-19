import numpy
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import time
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


train_time_ncv = 0
test_time_ncv = 0
train_time_cv = 0
test_time_cv = 0


def result_recordings(actual, preds):
    """
    Used to generate the confusion matrix
    :param actual: Actual Label values
    :param preds: Prediction vector
    :return: the confusion matrix
    """
    conf_matrix = numpy.zeros((2, 2), dtype=int)
    for elements in range(0, actual.shape[0]):
        if actual[elements] == 0 and preds[elements] == 0:
            conf_matrix[0, 0] = conf_matrix[0, 0] + 1
        elif actual[elements] == 1 and preds[elements] == 0:
            conf_matrix[0, 1] = conf_matrix[0, 1] + 1
        elif actual[elements] == 0 and preds[elements] == 1:
            conf_matrix[1, 0] = conf_matrix[1, 0] + 1
        else:
            conf_matrix[1, 1] = conf_matrix[1, 1] + 1

    return conf_matrix


def true_false_positive_negative(actual, preds, decision_boundary):
    """
    This function is used to calculate the true positive rate and false positive rate to be used in plotting ROC curve
    :param actual: Actual O/P labels
    :param preds: Predicted O/P labels
    :param decision_boundary: Decision threshold used to calculate which point is to be classified into which class
    :return: true positive rate and false positive rate
    """
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for elements in range(0, actual.shape[0]):
        if actual[elements] == 1 and preds[elements] > decision_boundary:
            true_positives += 1
        elif actual[elements] == 0 and preds[elements] > decision_boundary:
            false_positives += 1
        elif actual[elements] == 1 and preds[elements] <= decision_boundary:
            false_negatives += 1
        else:
            true_negatives += 1

    tpr = true_positives / (true_positives + false_negatives)
    fpr = false_positives / (false_positives + true_negatives)

    return tpr, fpr


def roc_curve_implementation(x_test, y_test, final_model, decision_boundary):
    """
    This function implements ROC curve, both the self implemented one and the one provided by sklearn
    :param x_test: test data points
    :param y_test: O/P labels for the test data points
    :param final_model: Cross validated model instance
    :param decision_boundary: Number of thresholds to try between 0 and 1
    """

    """
    - Since there is no decision boundary/decision function involved in naive bayes, we use probability for each class
    to help calculate all the true positive/false positive etc values
    - Initialised empty array to add the true positive rate and false positive rate values
    """
    pred_probs = final_model.predict_proba(x_test)[:, 1]
    roc = numpy.array([])

    """
    Iterate over decision boundary from 0 to 1 with steps of thousands, saving the tpr and fpr as we go for each
    """
    tpr_for_all_thresholds = []
    fpr_for_all_thresholds = []
    for db in numpy.linspace(0, 1, decision_boundary):
        tpr, fpr = true_false_positive_negative(y_test, pred_probs, db)
        tpr_for_all_thresholds.append(tpr)
        fpr_for_all_thresholds.append(fpr)
        roc = numpy.append(roc, [fpr, tpr])

    roc = roc.reshape(-1, 2)

    plt.figure(figsize=(15, 7))

    """
    sklearn implementation of ROC curve
    """
    fpr_sk, tpr_sk, thresholds = roc_curve(y_test, pred_probs)
    plt.plot(fpr_sk, tpr_sk, alpha=0.5, linewidth=4, color="blue", label="Scikit-learn")

    """
    Plotting the self implemented tpr and fpr data
    """
    plt.plot(roc[:, 0], roc[:, 1], color="red", alpha=0.5, linewidth=4, label="Our implementation")

    """
    Adding labels and legends
    """
    plt.title('ROC Curve', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.legend(loc='best')
    plt.show()

    """
    Area under the ROC curve - AUC
    """
    tpr_for_all_thresholds.reverse()
    fpr_for_all_thresholds.reverse()
    au_score = numpy.trapz(tpr_for_all_thresholds, fpr_for_all_thresholds)
    print("\nArea under the ROC curve is:\n", au_score)


def nb_ncv_eval(x_train, x_test, y_train, y_test, nb_ncv_instance):
    """
    This function evaluates classifier performance when cross validation is not performed
    :param x_train: training data
    :param x_test: testing data
    :param y_train: training data's labels
    :param y_test: testing data's labels
    :param nb_ncv_instance: classifier instance that hasn't been cross-validated
    :return:
    """
    start_train_time_ncv = time.time()
    nb_ncv_instance.fit(x_train, y_train)
    end_train_time_ncv = time.time()
    global train_time_ncv
    train_time_ncv = end_train_time_ncv - start_train_time_ncv

    start_test_time_ncv = time.time()
    prediction = nb_ncv_instance.predict(x_test)
    end_test_time_ncv = time.time()
    global test_time_ncv
    test_time_ncv = end_test_time_ncv - start_test_time_ncv

    error_num = sum(prediction != y_test)
    error_pc = (error_num / y_test.shape[0]) * 100
    print(
        "Naive Bayes implementation in sklearn without cross validation gives a model that yields an error percentage of"
        " : \n", error_pc, "\n")

    print("Confusion Matrix for the Non Cross Validated Data")
    print(result_recordings(y_test, prediction))


def nb_cv_eval(train_set, train_label, nb_cv_instance):
    """
    This function evaluates classifier performance when the model has been cross - validated
    :param train_set: training set data
    :param train_label: training set labels
    :param nb_cv_instance: cross - validated model's instance
    :return: Naive Bayes cross validated model instance
    """

    print("Cross Validation begins.....")
    skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    counter = 1
    final_data = []
    for train_index, test_index in skf.split(train_set, train_label):
        cv_train_data = train_set[train_index]
        cv_train_label = train_label[train_index]
        cv_test_data = train_set[test_index]
        cv_test_label = train_label[test_index]

        nb_cv_instance.fit(cv_train_data, cv_train_label)
        prediction = nb_cv_instance.predict(cv_test_data)
        error_num = sum(prediction != cv_test_label)
        error_pc = (error_num / cv_test_label.shape[0]) * 100
        final_data.append(error_pc)
        print("Score on fold ", counter, " is \n", error_pc)
        counter += 1

    print("\nMean of errors over 10 fold model training is : ", statistics.mean(final_data))
    print("Standard Deviation over 10 fold model training is : ", statistics.stdev(final_data))
    return nb_cv_instance


def naive_bayes_implementations(features, class_op, nbayes_cval, nbayes_non_cval):
    """
    This function implements Naive Bayes classification
    Both Cross validated models results and the non cross-validated model's results have been shown
    Both self implementation of ROC curve and sklearn's implementation of ROC curve has been shown
    GaussianNB is used since we consider the data to be normally distributed and continuous
    :param features: Features of data
    :param class_op: O/P labels or classes for the given data
    :param nbayes_cval: Cross Validation instance of naive bayes
    :param nbayes_non_cval: Non - cross validated instance of Gaussian Naive bias
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(features, class_op, test_size=0.25, random_state=1)

    """
    - Non - Cross Validated model's training and testing
    """
    nb_ncv_eval(x_train, x_test, y_train, y_test, nbayes_non_cval)

    """
    - Cross Validated model's training and testing
    """
    start_train_time_cv = time.time()
    final_model = nb_cv_eval(x_train, y_train, nbayes_cval)
    end_train_time_cv = time.time()
    global train_time_cv
    train_time_cv = end_train_time_cv - start_train_time_cv

    start_test_time_cv = time.time()
    prediction = final_model.predict(x_test)
    end_test_time_cv = time.time()
    global test_time_cv
    test_time_cv = end_test_time_cv - start_test_time_cv

    error_num = sum(prediction != y_test)
    error_pc = (error_num / y_test.shape[0]) * 100
    print("\nFinal error that the cross validated model of Naive Bayes classifier yields is : \n", error_pc)

    """
    - Calculation of Confusion Matrix on the predictions
    """
    confusion_matrix = result_recordings(y_test, prediction)
    print("\nCONFUSION MATRIX FOR THE TEST DATA: \n", confusion_matrix)

    roc_curve_implementation(x_test, y_test, final_model, 1000)


X_dat = pd.read_csv("messidor_features.csv",
                    names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13',
                           'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20'])
label = numpy.array(X_dat['x20'])

X_dat.drop(columns=['x20'], inplace=True)


def normalize(column):
    for i in column:
        X_dat[i] = X_dat[i] - X_dat[i].mean()
        X_dat[i] = X_dat[i] / X_dat[i].std()


X_dat = pd.read_csv("messidor_features.csv",
                    names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14',
                           'x15', 'x16', 'x17', 'x18', 'x19', 'x20'])
X_dat.drop(columns=['x20'], inplace=True)
normalize(['x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18'])

nb_cv = GaussianNB()
nb_ncv = GaussianNB()
naive_bayes_implementations(numpy.array(X_dat), label, nb_cv, nb_ncv)
print("\nTraining time of non cross validated model is \n", train_time_ncv, "\nTesting time of non cross validated model is \n", test_time_ncv)
print("\nTraining time of cross validated model is \n", train_time_cv, "\nTesting time of cross validated model is \n", test_time_cv)
