import time
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

train_time_ncv_lin = 0
test_time_ncv_lin = 0
train_time_cv_lin = 0
test_time_cv_lin = 0

train_time_ncv_rbf = 0
test_time_ncv_rbf = 0
train_time_cv_rbf = 0
test_time_cv_rbf = 0


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


def true_false_positive_negative(actual, preds):
    """
    Calculates all positives and all negatives to help plot the ROC curve
    :param actual:
    :param preds:
    :return: count of all positive and all negative values based on predictions
    """
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for elements in range(0, actual.shape[0]):
        if actual[elements] == 1 and preds[elements] == 1:
            true_positives += 1
        elif actual[elements] == 0 and preds[elements] == 1:
            false_positives += 1
        elif actual[elements] == 1 and preds[elements] == 0:
            false_negatives += 1
        else:
            true_negatives += 1

    all_positives = true_positives + false_negatives
    all_negatives = false_positives + true_negatives

    return all_positives, all_negatives


def roc_curve_implementation(final_model, x_test, y_test):
    """
    This function implements ROC curve, both the self implemented one and the one provided by sklearn
    :param x_test: test data points
    :param y_test: O/P labels for the test data points
    :param final_model: Cross validated model instance
    """

    """
    - Since the prediction is based on the decision boundary in the classifier we can use the decision_function to get the
     scores  
    """
    true_positives = 0
    false_positives = 0
    predictions = final_model.predict(x_test)
    all_positives, all_negatives = true_false_positive_negative(y_test, predictions)

    scores = final_model.decision_function(x_test)

    sort_index = scores.argsort()[::-1]
    scores_sorted = scores[sort_index]
    y_test_sorted = y_test[sort_index]

    point = numpy.empty((0, 2))
    f_prev = float('-inf')

    """
    - Append points as you go for each data point, the tpr and fpr
    - If the score is  same as the previous one, the entry is skipped
    """
    point = numpy.append(point, ([[0, 0]]), axis=0)
    for score, lab in zip(scores_sorted, y_test_sorted):
        if score != f_prev:
            if score > 0 and lab == 1:
                true_positives += 1
            elif score > 0 and lab == 0:
                false_positives += 1
            else:
                pass
            point = numpy.append(point, ([[(false_positives/all_negatives), (true_positives/all_positives)]]), axis=0)
        else:
            pass
    point = numpy.append(point, numpy.array([[1, 1]]), axis=0)

    """
    sklearn implementation of ROC curve
    """
    fpr_sk, tpr_sk, thresholds = roc_curve(y_test, scores)
    plt.plot(fpr_sk, tpr_sk, alpha=0.5, color="blue", linewidth=4, label="Scikit-learn")

    """
    Plotting the self implemented tpr and fpr data
    """
    plt.plot(point[:, 0], point[:, 1], color="red", alpha=0.5, linewidth=4, label="Our implementation")

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
    tpr_for_all_thresholds = point[:, 1]
    fpr_for_all_thresholds = point[:, 0]
    au_score = numpy.trapz(tpr_for_all_thresholds, fpr_for_all_thresholds)
    print("\nArea under the ROC curve is:\n", au_score)


def ncv_kernel_evaluator(features, class_op):
    """
    This function evaluates the classifier performance when the model hasn't been cross-validated or parameters tuned
    :param features: Features of the data set
    :param class_op: Class labels for the given data points
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(features, class_op, test_size=0.25, random_state=1)
    clf_lin = SVC(kernel='linear', random_state=1)
    start_train_lin = time.time()
    clf_lin.fit(x_train, y_train)
    end_train_lin = time.time()
    global train_time_ncv_lin
    train_time_ncv_lin = end_train_lin - start_train_lin

    start_test_lin = time.time()
    predict_lin = clf_lin.predict(x_test)
    end_test_lin = time.time()
    global test_time_ncv_lin
    test_time_ncv_lin = end_test_lin - start_test_lin

    error_num_lin = sum(predict_lin != y_test)
    error_pc_lin = (error_num_lin / y_test.shape[0]) * 100
    print(
        "SVM implementation in sklearn for linear kernel without cross validation gives a model that yields an error "
        "percentage of : \n", error_pc_lin, "\n")

    clf_rbf = SVC(kernel='rbf', random_state=1)
    start_train_rbf = time.time()
    clf_rbf.fit(x_train, y_train)
    end_train_rbf = time.time()
    global train_time_ncv_rbf
    train_time_ncv_rbf = end_train_rbf - start_train_rbf

    start_test_rbf = time.time()
    predict_rbf = clf_rbf.predict(x_test)
    end_test_rbf = time.time()
    global test_time_ncv_rbf
    test_time_ncv_rbf = end_test_rbf - start_test_rbf

    error_num_rbf = sum(predict_rbf != y_test)
    error_pc_rbf = (error_num_rbf / y_test.shape[0]) * 100
    print(
        "SVM implementation in sklearn for rbf kernel without cross validation gives a model that yields an error "
        "percentage of : \n", error_pc_rbf, "\n")

    if error_pc_lin > error_pc_rbf:
        print("Non Cross validated model does better on the RBF Kernel")
        print(result_recordings(y_test, predict_rbf))
    else:
        print("\nNon Cross validated model does better on the Linear Kernel\n")
        print("Confusion Matrix for Non-cross validated model is:")
        print(result_recordings(y_test, predict_lin), "\n")


def cv_kernel_evaluator(features, class_op):
    """

    :param features:
    :param class_op:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(features, class_op, test_size=0.25, random_state=1)

    clf_lin = SVC(kernel='linear', random_state=1)

    print("Cross validation for Linear Kernel begins.....")
    start_train_lin = time.time()
    c_range_lin = [0.1, 1, 10, 100, 1000]
    parameter_grid_lin = dict(C=c_range_lin)
    grid_lin = GridSearchCV(clf_lin, param_grid=parameter_grid_lin, cv=10, verbose=1)

    grid_lin.fit(x_train, y_train)
    end_train_lin = time.time()
    global train_time_cv_lin
    train_time_cv_lin = end_train_lin - start_train_lin

    start_test_lin = time.time()
    predict_lin = grid_lin.predict(x_test)
    end_test_lin = time.time()
    global test_time_cv_lin
    test_time_cv_lin = end_test_lin - start_test_lin

    error_num_lin = sum(predict_lin != y_test)
    error_pc_lin = (error_num_lin / y_test.shape[0]) * 100
    print(
        "SVM implementation in sklearn for Linear kernel with cross validation and C value", grid_lin.best_params_['C'],
        "gives a model that yields an error percentage of : \n", error_pc_lin, "\n")

    """
    RBF Kernel Cross-Validation
    """
    clf_rbf = SVC(kernel='rbf', random_state=1)
    print("Cross validation for RBF Kernel begins.....")
    start_train_rbf = time.time()
    c_range_rbf = [0.1, 1, 10, 100, 1000]
    gamma_rbf = [0.1, 1, 10, 100]
    parameter_grid_rbf = dict(C=c_range_rbf, gamma=gamma_rbf)
    grid_rbf = GridSearchCV(clf_rbf, param_grid=parameter_grid_rbf, cv=10, verbose=1)
    grid_rbf.fit(x_train, y_train)
    end_train_rbf = time.time()
    global train_time_cv_rbf
    train_time_cv_rbf = end_train_rbf - start_train_rbf

    start_test_rbf = time.time()
    predict_rbf = grid_rbf.predict(x_test)
    end_test_rbf = time.time()
    global test_time_cv_rbf
    test_time_cv_rbf = end_test_rbf - start_test_rbf

    error_num_rbf = sum(predict_rbf != y_test)
    error_pc_rbf = (error_num_rbf / y_test.shape[0]) * 100
    print(
        "SVM implementation in sklearn for RBF kernel with cross validation and C value", grid_rbf.best_params_['C'],
        ", gamma/spread value of, ", grid_rbf.best_params_['gamma'], "gives a model that yields an error percentage of :"
        " \n", error_pc_rbf, "\n")

    if error_pc_rbf < error_pc_lin:
        confusion_matrix = result_recordings(y_test, predict_rbf)
        print("\nCONFUSION MATRIX FOR THE TEST DATA: \n", confusion_matrix)
        return grid_rbf, x_test, y_test
    else:
        confusion_matrix = result_recordings(y_test, predict_lin)
        print("\nCONFUSION MATRIX FOR THE TEST DATA: \n", confusion_matrix)
        return grid_lin, x_test, y_test


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

ncv_kernel_evaluator(numpy.array(X_dat), label)
model, test_data, test_labels = cv_kernel_evaluator(numpy.array(X_dat), label)
roc_curve_implementation(model, test_data, test_labels)

print("\nTraining time of non cross validated model (linear kernel) is \n", train_time_ncv_lin, "\nTesting time of non cross validated model (linear kernel) is \n", test_time_ncv_lin)
print("\nTraining time of non cross validated model (rbf kernel) is \n", train_time_ncv_rbf, "\nTesting time of non cross validated model (rbf kernel) is \n", test_time_ncv_rbf)

print("\nTraining time of cross validated model (linear kernel) is \n", train_time_cv_lin, "\nTesting time of cross validated model (linear kernel) is \n", test_time_cv_lin)
print("\nTraining time of cross validated model (rbf kernel) is \n", train_time_cv_rbf, "\nTesting time of cross validated model (rbf kernel) is \n", test_time_cv_rbf)
