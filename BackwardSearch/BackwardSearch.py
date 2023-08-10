'''We will generate a random covariance matrix for each of our 2 classes. With this matrix we will generate a dataset for each of the class, with each class having 10
attributes and 1000 rows. We then combine the data and split it into Train and Test sets. We calculate the error percentage of the original data set. We then remove 1
column from the data set one by one and then calculate the error percentage with the one column removed. We then remove the column which caused the least error from the
original dataset. We redo this process again on the reduced dataset till we find the 5 best attributes among the 10.'''

#Importing Libraries
import numpy as np
from sklearn import datasets
import sklearn.discriminant_analysis
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#Means for each of the 10 attributes of the 2 classes.
mean1 = [4,5,4,5,4,5,4,5,4,5]
mean2 = [-4,-5,-4,-5,-4,-5,-4,-5,-4,-5]

#Generating Covariance matrices for the 2 classes
np.random.seed(42)
covpre1 = 5 + 40 * np.random.randn(10, 10)
cov1 = (np.dot(covpre1, covpre1.transpose()))
np.random.seed(42)
covpre2 = 20 + 20 * np.random.randn(10, 10)
cov2 = (np.dot(covpre2, covpre2.transpose()))

#Generating randon data from the covariance matrices
np.random.seed(42)
x1 = np.random.multivariate_normal(mean1,cov1, 1000)
np.random.seed(42)
x2 = np.random.multivariate_normal(mean2,cov2, 1000)

#Combining the Data and class values for the 2 classes
X = np.concatenate((x1,x2))
Xc = np.ones(1000)
Xc = np.concatenate((Xc, np.zeros(1000)))

#Splitting Data into Training and Testing Set with a 80:20 Split
XTrain, XTest , XcTrain, XcTest = train_test_split(X,Xc, test_size=0.2, stratify=Xc)

#Calculating Error Percentage for the original data
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis() #Creating Object for LDA
lda.fit(XTrain,XcTrain) #Fitting the LDA Model
prediction = lda.predict(XTest) #Performing Prediction on the model 
ogError = sum(abs(prediction - XcTest)) #Calculating number of errors
ogClassError = (ogError/XTest.shape[0]) * 100 #Calculating percentage of errors

#Creating a list to store the error rates before and after each dimension reduction
classError = []
classError.append(ogClassError)

#We have a data set with 10 attributes. This loop will iterate through 5 reductions in dimensions. During each loop it will first create a list(subClassError) to store the error rates of the test data after removing each attribute.
#After this another loop is run to perform predictions after removing each attribute(column). In the inner loop, we first delete the first column and find out the error rate 
#for the classification with the remaining columns. This error rate is stored in the subClassError list. The same thing is done after removing the 2nd, 3rd coulmn and so on. 
#Then the column whose removal caused the lowest classification error is ejected from the data set. Now we have a data set with the 9 best attributes. We perform the same process
#with the remaining attributes 4 more times.
for i in range(5):
    subClassError = [] #List to store error rates after removing each attribute
    for col in range(X.shape[1]):
        XNew = np.delete(X,col,1) #Deleting Column
        XTrain, XTest , XcTrain, XcTest = train_test_split(XNew,Xc, test_size=0.2, stratify=Xc) #Splitting the reduced dataset into train test set
        lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis() #Creating Object for LDA
        lda.fit(XTrain,XcTrain) #Fitting the LDA Model
        prediction = lda.predict(XTest)#Performing Prediction on the model 
        error = sum(abs(prediction - XcTest)) #Calculating number of errors
        subClassError.append(error/XTest.shape[0] * 100) #Calculating percentage of errors
    accuracyLstSorted = np.argsort(subClassError) #Sorting the error list in ascending order. This way we get the column whose removal caused the lowest classification error at first
    classError.append(subClassError[accuracyLstSorted[0]]) #Adding the best classification rate to the list
    X = np.delete(X,accuracyLstSorted[0],1) #Deleting the column whose removal caused the lowest classification error
    print(classError)

#Plotting the Classification Error rate for each reduction in dimension
plt.plot(range(1,7), classError, label = 'Classification Error(%)')
plt.ylim([0,100])
plt.legend()
plt.show()
