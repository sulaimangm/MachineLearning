'''
Here we are trying out the implementation of Naive Bayes Classifier. We are using a text file attached 'BayesTrainData' and 'BayesTestData' for the test and train sets 
respectively. We are first importing the data and performing preprocessing steps on it, before splitting the training data into a different public and private set. We 
then calculate the mean and standard deviation for each attribute in the private and public class seperately. We then import the Testing data and preprocess it. Then 
we traverse the test data calculation the probabilities of each element in each row and then calculate the probability of it being in one class or the other by multiplying
all the individual probabilities.
'''

#Importing Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn.discriminant_analysis
import math

# Importing Training Data and preprocessing it
TrainData = pd.read_csv("BayesTrainData.csv") # Import
Train = TrainData.drop(columns=['University Name']) # Dropping University Name Column
XTrainwc = Train.iloc[:,:].values   #Removing Headers
XTrain = Train.iloc[:,[0,2,3,4,5,6]].values #Data without class
XcTrain = Train.iloc[:,1].values    #Data with Class

#Splitting the data into Public and Private set
XwcPublic, XwcPrivate = [x for _, x in Train.groupby(Train['Private/Public'] == 'private')] #Split
public = XwcPublic.iloc[:,[0,2,3,4,5,6]].values #Public Data without class
private = XwcPrivate.iloc[:,[0,2,3,4,5,6]].values # Private Data without class

#Calculating the mean and standard deviation for the SAT verbal and SAT math column for both the public and private dataset
verbalMeanPrivate = XwcPrivate["SAT verbal"].mean()
verbalStdPrivate = XwcPrivate["SAT verbal"].std()
verbalMeanPublic = XwcPublic["SAT verbal"].mean()
verbalStdPublic = XwcPublic["SAT verbal"].std()
mathMeanPrivate = XwcPrivate["SAT math"].mean()
mathStdPrivate = XwcPrivate["SAT math"].std()
mathMeanPublic = XwcPublic["SAT math"].mean()
mathStdPublic = XwcPublic["SAT math"].std()

#print(verbalMeanPrivate,verbalMeanPublic,mathMeanPrivate,mathMeanPublic)
#print(verbalStdPrivate,verbalStdPublic,mathStdPrivate,mathStdPublic)

# Importing Testing Data and preprocessing it
TestData = pd.read_csv("BayesTestData.csv") # Import
Test = TestData.drop(columns=['University Name'])   # Dropping University Name Column
XTestwc = Test.iloc[:,:].values #Removing Headers
XTest = Test.iloc[:,[0,2,3,4,5,6]].values   #Data without class
XcTest = Test.iloc[:,1].values  #Data with Class

# Function to calculate probabilities for Discrete Attributes with a and be being row and column variables
#This function will calculate the probability of an attribute given it is public and private
def DiscreteProbablity(a,b):
    count = 0    
    for k in private:
        if(k[b]==XTest[a][b]):
            count += 1
    privateProb.append(count/len(private))
    count = 0
    for k in public:
        if(k[b]==XTest[a][b]):
            count += 1
    publicProb.append(count/len(public))

prediction = []

# This loop traverses through the Test Data row by row, column by column checking for the probabilities of
# every data in each row. It will store the all the marginal probablities of the discrete attributes and 
# probability distribution function of continous attributes. It will then multiply the lists of private and 
# public probablities separately and then compare them. It will then store 'private' if the probability of 
# data given private is larger or it will store 'public' if the probability of data given public is larger.
# If both the probablities are uncomparable i.e. 0, it will store 'unknown'.  
for i in range(len(XTest)):
    privateProb = []
    publicProb =[]
    for j in range(len(XTest[i])):
        if(j == 0):
            DiscreteProbablity(i,j)

        if(j == 1):
            pri = (1/math.sqrt(2*math.pi*(verbalStdPrivate**2)))*math.exp((-1/(2*(verbalStdPrivate**2)))*((XTest[i][j]-verbalMeanPrivate)**2))
            privateProb.append(pri)
            pub = (1/math.sqrt(2*math.pi*(verbalStdPublic**2)))*math.exp((-1/(2*(verbalStdPublic**2)))*((XTest[i][j]-verbalMeanPublic)**2))
            publicProb.append(pub)
        
        if(j == 2):
            pri = (1/math.sqrt(2*math.pi*(mathStdPrivate**2)))*math.exp((-1/(2*(mathStdPrivate**2)))*((XTest[i][j]-mathMeanPrivate)**2))
            privateProb.append(pri)
            pub = (1/math.sqrt(2*math.pi*(mathStdPublic**2)))*math.exp((-1/(2*(mathStdPublic**2)))*((XTest[i][j]-mathMeanPublic)**2))
            publicProb.append(pub)
                
        if(j == 3):
            DiscreteProbablity(i,j)
            
        if(j == 4):
            DiscreteProbablity(i,j)

        if(j == 5):
            DiscreteProbablity(i,j)

    privateProb.append(len(private)/len(XTrainwc))
    publicProb.append(len(public)/len(XTrainwc))
    # print("Private: ",privateProb)
    # print("Public: ",publicProb)
    privateProbProduct = 1
    for z in privateProb:
        privateProbProduct *= z
    publicProbProduct = 1
    for z in publicProb:
        publicProbProduct *= z
    print(f"Probability for (row {i+1}|Private): {privateProbProduct}")
    print(f"Probability for (row {i+1}|Public): {publicProbProduct}")
    if(privateProbProduct>publicProbProduct):
        prediction.append("private")
        print("Probablity for (Data|Private) is greater\n")
    elif(publicProbProduct>privateProbProduct):
        prediction.append("public")
        print("Probablity for (Data|Public) is greater\n")
    else:
        prediction.append("unknown")
        print("Both Probabilities are 0. Unable to compare\n")

# Confusion Matrix
# True Negative - Prediction and Actual are public
# True Positive - Prediction and Actual are private
# Unknown - If both the probabilities are 0
tp=0
fp=0
tn=0
fn=0
unknown = 0
for i in range(0,len(prediction)):
    if (XcTest[i]== 'public' and prediction[i]== 'public'):
        tn=tn+1
    elif (XcTest[i]== 'private' and prediction[i]== 'private'):
        tp=tp+1
    elif (XcTest[i]== 'public' and prediction[i]== 'private'):
        fp=fp+1
    elif (XcTest[i]== 'private' and prediction[i]== 'public'):
        fn=fn+1
    else:
        unknown += 1
print("Confusion Matrix:")
print(pd.DataFrame([[tn,fp],[fn,tp]], columns = [0,1]))
print(f"Unable to classify {unknown} row as both probablities are 0.")
