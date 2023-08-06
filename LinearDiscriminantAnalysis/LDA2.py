'''
Here we are trying out the implementation of Linear Discriminant Analysis(LDA). We are using a text file attached 'spam.csv'. We ar eimporting it and dividing the
dataset into the 2 classes while also seperating the data and the class coulmns. Then we are calculating the best projection line. Over this projection line, keeping
threshold as 0, we are calculating the accuracy. Over this predication results we are calculating the confusion matrix. Then we run a loop to find the best threshold 
and calculate the accuracy with the best value of threshold. Over this predication results we again calculate the confusion matrix.
'''

#Importing Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn.discriminant_analysis

# Computing Confusion Matrix
def ConfMat(actual, pred):
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(0,len(pred)):
        if (Xc[i]==0 and pred[i]==0):
            tn=tn+1
        elif (Xc[i]==1 and pred[i]==1):
            tp=tp+1
        elif (Xc[i]==0 and pred[i]==1):
            fp=fp+1
        elif (Xc[i]==1 and pred[i]==0):
            fn=fn+1
    print(pd.DataFrame([[tn,fp],[fn,tp]], columns = [0,1]))
    print("\nNumber of Data Points in the first class classified correctly out of 500: ",tp)
    print("Number of Data Points in the second class classified correctly out of 500: ",tn)

Xwc = pd.read_csv("spam.csv") # Importing DataSet

# Dividing DataSet into Data and Class
X = Xwc.iloc[:,0:57].values
Xc = Xwc.iloc[:,57].values

# Dividing Dataset into 2 classes
X2wc, X1wc = [x for _, x in Xwc.groupby(Xwc['Column58'] == 1)]

# Seperating the Divided DataSet into 2 classes
X1 = X1wc.iloc[:,0:57].values
X2 = X2wc.iloc[:,0:57].values

# Mean Centering the data
mean1 = np.mean(X1,0)
mean2 = np.mean(X2,0)
X1mc = X1 - mean1
X2mc = X2 - mean2

# Determining the Discriminant Line found by FLD
S1 = np.dot(X1mc.T, X1mc)
S2 = np.dot(X2mc.T, X2mc)
Sw = S1 + S2
w = np.dot(np.linalg.inv(Sw),(mean1 - mean2))

# Prediction while keeping threshold as 0
thresh = 0
prediction = (np.sign(np.dot(w,X.T)+thresh)+1)/2
error = 100*sum(prediction != Xc)/X.shape[0]
temperror = error
tempthresh = thresh

print("\nPercentage of DataPoints incorrectly classified before setting Threshold = ", error)

print("\nConfusion Matrix Before Setting Threshold:")
ConfMat(Xc,prediction)
print("As we can see above, almost all of the datapoints in the first class are classified correctly while many from the second class are classified incorrectly\n")

# Computing the threshold with the least error for 100 different thresholds
i = 0
while (i<100):
    thresh-=0.0001
    prediction = (np.sign(np.dot(w,X.T)+thresh)+1)/2
    error = 100*sum(prediction != Xc)/X.shape[0]
    if (error<temperror):
        temperror = error
        tempthresh = thresh
    i+=1
    #print(thresh,error)
thresh = tempthresh
print(f"We get the lowest error as {temperror} at threshold {tempthresh}.")

# Prediction after setting the best threshold
prediction = (np.sign(np.dot(w,X.T)+thresh)+1)/2
error = 100*sum(prediction != Xc)/X.shape[0]

# Confusion matrix after setting threshold
print("Percentage of DataPoints incorrectly classified after setting Threshold = ", error)
print("\nConfusion Matrix After Setting Threshold:")
ConfMat(Xc,prediction)
