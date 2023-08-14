'''
In this Code, we are performing Classifier Performance evaluation after applying PCA. The 2 classifiers we will be using for evaluation are Linear Discriminant Analysis(LDA)
and Random Forest Classifier(RDC). We will import the data and perform preprocessing steps on it. We will then reduce the dimensions of the data one by one while bringing
it into the Principal Component space. On the reduced data we will perform classification using both LDA and RDC. During the classification process, for evaluation we 
will calculate the Training Time, Testing Time, Scores(Accuracy) and the Confusion Matrix for each Dimension reduced.
'''

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import time

#Mean Centering and scaling the data to unit variance
def normalize(column):
    for i in column:
        X_dat[i]=X_dat[i]-X_dat[i].mean()
        X_dat[i]=X_dat[i]/X_dat[i].std()

#Function for generating Confusion Matrix 
def confmat(y_preds):
    tp=0
    fp=0
    tn=0
    fn=0
    for k in range(0,len(y_preds)):
        if (Y_test.iloc[k]==0 and y_preds[k]==0):
            tn=tn+1
        elif (Y_test.iloc[k]==1 and y_preds[k]==1):
            tp=tp+1
        elif (Y_test.iloc[k]==0 and y_preds[k]==1):
            fp=fp+1
        elif (Y_test.iloc[k]==1 and y_preds[k]==0):
            fn=fn+1
    return np.array([[tn,fp],[fn,tp]])         

#Creating Objects for classifiers
LDA=LinearDiscriminantAnalysis()
rdf_clf=RandomForestClassifier()

#Importing and Formating the data
X_dat=pd.read_csv("messidor_features.csv",names=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20'])
X_dat.head() #Printing first few rows of the dataset
Y=X_dat['x20'] #Assigning last column as the class values
X_dat.drop(columns=['x20'],inplace=True) #Dropping the class values from the data set
normalize(['x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18']) #Function call to mean-center and scale the data

#Splitting Data into Training and Testing Set with a 75:25 Split
np.random.seed(42)
X_train,X_test,Y_train,Y_test=train_test_split(X_dat,Y,test_size=0.25)


#Calculating the eigen values and vectors and sorting eigen vectors wrt the descending order of the eigen values
eigval,eigvec=np.linalg.eig(X_train.transpose()@X_train)
ind=np.argsort(eigval)
ind=ind[::-1]
pd.DataFrame(eigvec[0:,ind[0:]])


#Calculating the mean squared error after reconstructing the data from reduced dimensions
mae=[mean_squared_error(X_dat,X_dat@(eigvec[0:,ind[0:19-i]]@eigvec[0:,ind[0:19-i]].transpose())) for i in range(0,19)]
print("Mean Squared error After removing:")
print()
for i in range(0,len(mae)):
    print(f" {i} components \t {round(mae[i], 3)}")
print()

#Initializing variables for Scores, Training Time, Testing Time and Confusion Matrix
rdf_scores=np.array([])
lda_scores=np.array([])
lda_train_time=np.array([])
rdf_train_time=np.array([])
lda_test_time=np.array([])
rdf_test_time=np.array([])
lda_confmat=[]
rdf_confmat=[]


#Calculating the accuracy from LDA and Randomforest using PCA along with computing testing and training time
for i in range(0,19):
    X_train1=X_train@(eigvec[0:,ind[0:19-i]]) #Bring the training data into the principal component space while reducing dimensions
    
    #Training Time for LDA
    start = time.time()
    LDA.fit(X_train1,Y_train)
    stop = time.time()
    lda_train_time=np.append(lda_train_time,stop-start)
    
    #Training Time for Random Forest
    start = time.time()
    rdf_clf.fit(X_train1,Y_train)
    stop = time.time()
    rdf_train_time=np.append(rdf_train_time,stop-start)
    
    #Testing Time for LDA
    start=time.time()
    y_preds=LDA.predict(X_test@(eigvec[0:,ind[0:19-i]])) #Bring the testing data into the principal component space while reducing dimensions
    stop=time.time()
    lda_test_time=np.append(lda_test_time,stop-start)
    lda_scores=np.append(lda_scores,accuracy_score(Y_test,y_preds)) #Calculating Score
    lda_confmat.append(pd.DataFrame(confmat(y_preds))) #Calculating Confusion Matrix
    
    #Testing Time for Random Forest
    start=time.time()
    y_preds=rdf_clf.predict(X_test@(eigvec[0:,ind[0:19-i]]))
    stop=time.time()
    rdf_test_time=np.append(rdf_test_time,stop-start)
    rdf_scores=np.append(rdf_scores,accuracy_score(Y_test,y_preds)) #Calculating Score
    rdf_confmat.append(pd.DataFrame(confmat(y_preds))) #Calculating Confusion Matrix

#Final Results
scores=pd.DataFrame(zip(lda_scores,rdf_scores),columns=['LDA','RandomForest'])
print("Comparison of accuracy between LDA and Random Forest Classifer ")
print()
print(scores)

print()
print("Training Time comparison between LDA and RandomForest Classifier ")
print()
print(pd.DataFrame(zip(lda_train_time,rdf_train_time),columns=['LDA','RandomForest']))

print()
print("Testing Time comparison between LDA and RandomForest Classifier ")
print()
print(pd.DataFrame(zip(lda_test_time,rdf_test_time),columns=['LDA','RandomForest']))

print()
print("Confusion Matrices for Linear Discriminator Classifier")
print()
for i in range(0,len(lda_confmat)):
    print(f"After reducing {i} dimensions")
    print(pd.DataFrame(lda_confmat[i]))
    print()

print()
print("Confusion Matrices for Random Forest Classifier")
print()
for i in range(0,len(lda_confmat)):
    print(f"After reducing {i} dimensions")
    print(pd.DataFrame(rdf_confmat[i]))
    print()