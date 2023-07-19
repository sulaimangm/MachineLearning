import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import time

#pd.set_option("display.max_rows", None, "display.max_columns", None)


'''
Mean Centering and scaling the data to unit variance
'''
def normalize(column):
    for i in column:
        X[i]=X[i]-X[i].mean()
        X[i]=X[i]/X[i].std()


'''
Computing Confusion Matrix
'''
def ConfMat(y_preds):
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(0,len(y_preds)):
        if (XcTest.iloc[i]==0 and y_preds[i]==0):
            tn=tn+1
        elif (XcTest.iloc[i]==1 and y_preds[i]==1):
            tp=tp+1
        elif (XcTest.iloc[i]==0 and y_preds[i]==1):
            fp=fp+1
        elif (XcTest.iloc[i]==1 and y_preds[i]==0):
            fn=fn+1
    print(pd.DataFrame([[tn,fn],[fp,tp]], columns = [0,1]))


'''
Importing and PreProcessing the Data
'''
X=pd.read_csv("messidor_features.csv",names=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20'])
Xc=X['x20'] #Assigning Target Class to a variable
X.drop(columns=['x20'],inplace=True)    #Dropping the Last column indication the Target Class
normalize(['x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18']) # Normalizing the Data


'''
Splitting Data into Test and Train Set with split set as 75% for Train and 25% for Test
'''
np.random.seed(42)
XTrain,XTest,XcTrain,XcTest=train_test_split(X,Xc,test_size=0.25)


'''
Performing Classification without Tuning parameters
'''
print("\nKNN Without Tuning Parameters:")
start = time.time()
knnNonTune = KNeighborsClassifier()
knnNonTune.fit(XTrain,XcTrain)
stop = time.time()
trainTime=stop-start
print(f'The Training Time is: {trainTime:.4f}s')
start = time.time()
print(f'The Accuracy of KNN Classifier without Tuning the Parameters is {knnNonTune.score(XTest,XcTest)*100}')
stop = time.time()
testTime=stop-start
print(f'The Testing Time is: {testTime:.4f}s')
prediction = knnNonTune.predict(XTest)
print("\nConfusion Matrix for the Testing Data before Parameter Tuning:")
ConfMat(prediction)


'''
Performing K-Fold Cross Validation while taking a random number of Neighbours
'''
print("\nKNN Using Cross-Validation:")
knnCV = KNeighborsClassifier(n_neighbors=20)    #Randomly setting neighbours as 20
cv_scores = cross_val_score(knnCV, XTrain, XcTrain, cv=5, scoring = 'accuracy')
print(f'The Accuracy Rates of the different folds are: {cv_scores*100}')
print(f'The Average Accuracy is: {np.mean(cv_scores)*100}')


'''
Performing K-Fold Cross Validation with Parameter Selection using Grid-Search
'''
print("\nKNN Using Grid-Search:")
start = time.time()
kList = list(range(1,100,2)) #Setting value of neighbours from 1 to 99 only taking the odd numbers to eliminate chances of tie
knn = KNeighborsClassifier()
params = {'n_neighbors' : kList} #Setting parameters for grid-search
grid_kn = GridSearchCV(estimator = knn, param_grid = params, scoring = 'accuracy', cv = 10, verbose = 1, n_jobs=-1) #10 fold Cross Validation
grid_kn.fit(XTrain, XcTrain)
stop = time.time()
trainTime=stop-start
print(f'The Training Time is: {trainTime:.4f}s')


'''
Plotting the graph of no. of neighbours vs Accuracy
'''
plt.title('Parameter vs Accuracy')
plt.plot(kList, grid_kn.cv_results_['mean_test_score'], 'b', label = 'K vs Accuracy')
plt.legend(loc = 'lower right')
plt.xlim([0, 100])
plt.ylabel('Accuracy(%)')
plt.xlabel('Parameter(k)')
plt.show()

#print(pd.DataFrame(grid_kn.cv_results_['mean_fit_time']))

print(f'The optimal number of neighbours to use for best accuracy is: {grid_kn.best_estimator_}')
print(f'Accuracy for the Training Set with the most optimal number of neighbours is: {grid_kn.best_score_*100}')


'''
Predicting Class Values of Test Set using the best Parameters
'''
start = time.time()
prediction = grid_kn.predict(XTest)
stop = time.time()
testTime=stop-start
print("\nPrediction on the Test Data using the best found Parameters:")
print(f'The Testing Time is: {testTime:.4f}s')
print(f'Accuracy for the Test Set with the most optimal number of neighbours is {grid_kn.score(XTest, XcTest)*100}')
print("\nConfusion Matrix for the Testing Data after Parameter Tuning:")
ConfMat(prediction)


'''
Plotting the ROC Corve using SKLearn.metrics built-in function
'''
XTestScores = grid_kn.predict_proba(XTest) # Fetching the probabilities of each of the data point being in a class
positiveScores = XTestScores[:,1]# Probabilities of the data point being in the Positive Class
fpr, tpr, threshold = roc_curve(XcTest, XTestScores[:, 1])  # Calculating False Positive Rate and True Positive Rate
roc_auc = auc(fpr, tpr) #Calculation Area Under Curve for the in-built function of the ROC Plot
plt.title('ROC Using Inbuilt Function')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


'''
Calculating False Positive and True Positive Rates
'''
y = XcTest.iloc[:].values
fpr = []    # false positive rate
tpr = []    # true positive rate
thresholds = np.arange(0.0, 1.01, .01)  # Setting Thresholds from 0.0, 0.01, ... 1.0
# get number of positive and negative examples in the dataset
P = sum(y)
N = len(y) - P
#Iterate through all thresholds and determine fraction of true positives and false positives found at each threshold
for thresh in thresholds:
    FP=0
    TP=0
    for i in range(len(positiveScores)):
        if (positiveScores[i] > thresh):
            if y[i] == 1:
                TP = TP + 1
            if y[i] == 0:
                FP = FP + 1
    fpr.append(FP/float(N))
    tpr.append(TP/float(P))


'''
Plotting Self Generated ROC Graph
'''
plt.title('Self Generated ROC')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
roc_auc = auc(fpr, tpr) #Calculation Area Under Curve for the Self implementation of the ROC Plot
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.show()