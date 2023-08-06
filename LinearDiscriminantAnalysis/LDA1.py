'''
Here we are trying out the implementation of Linear Discriminant Analysis(LDA). We are using a text file attached 'fld.txt'. We ar eimporting it and dividing the
dataset into the 2 classes while also seperating the data and the class coulmns. Then we are calculating the best projection line. Over this projection line, keeping
threshold as 0, we are calculating the accuracy. Then we run a loop to find the best threshold and calculate the accuracy with the best value of threshold. We then 
implement SKLearn's in built method for LDA and calculate the accuracy using that method. We then plot the graph showing the projection line and also the 3 different
discriminant lines we got from different methods.
'''

#Importing Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn.discriminant_analysis

Xwc = pd.read_csv("fld.txt", names = ['att1', 'att2', 'class']) # Importing DataSet

# Dividing DataSet into Data and Class
X = Xwc.iloc[:,[0,1]].values
Xc = Xwc.iloc[:,2].values

# Dividing Dataset into 2 classes
X2wc, X1wc = [x for _, x in Xwc.groupby(Xwc['class'] == 1)]

# Seperating the Divided DataSet into 2 classes
X1 = X1wc.iloc[:,[0,1]].values
X2 = X2wc.iloc[:,[0,1]].values

# Plotting the Data
plt.scatter(X1[:,0],X1[:,1], c = 'r', label = 'Class 1')
plt.scatter(X2[:,0],X2[:,1], c = 'b', label = 'Class 0')

# Mean Centering the data
mean1 = np.mean(X1,0)
mean2 = np.mean(X2,0)
X1mc = X1 - mean1
X2mc = X2 - mean2

#Determining the Discriminant Line found by FLD
S1 = np.dot(X1mc.T, X1mc)
S2 = np.dot(X2mc.T, X2mc)
Sw = S1 + S2
w = np.dot(np.linalg.inv(Sw),(mean1 - mean2))
print("\nW :",w)

plt.plot([-2000*w[0], 2000*w[0]], [-2000*w[1], 2000*w[1]], c = 'k', label = 'W') # Plotting W

thresh = 0 #Threshold is kept as 0

# Determining the Class of the Data Points
prediction = (np.sign(np.dot(w,X.T)+thresh)+1)/2
error = 100*sum(prediction != Xc)/X.shape[0]

print("\nError Percentage with In-Class Method keeping threshold as 0 = ", error) # Percentage of data points incorrectly classified

# Calculating Slope and YIntercept
slope = -w[0]/w[1]
y_int = -thresh/w[1]
print("Slope: ",slope)
print("Y_Intercept: ",y_int)

# Plotting the Discriminant line calculated using Manual Implementation
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = slope * x_vals + y_int
plt.plot(x_vals, y_vals, 'g--', label = 'Discriminant Line with 0 Thresh') # Plotting the Discriminant Line

# Computing the threshold with the least error for 100 different thresholds
temperror = error
tempthresh = thresh
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
print("\n After determining the best possible threshold:")
print(f"We get the lowest error as {temperror} at threshold {tempthresh}.")

#Calculating Slope and Intercept
slope = -w[0]/w[1]
y_int = -thresh/w[1]
print("Slope: ",slope)
print("Y_Intercept: ",y_int)

#Preprocessing for plotting the discriminant line
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = slope * x_vals + y_int
plt.plot(x_vals, y_vals, 'm--', label = 'Discriminant Line with best Thresh') # Plotting the Discriminant Line

# Classification using SKLearn's in-built Library for LDA
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(X,Xc)
prediction = lda.predict(X)
error = sum(abs(prediction - Xc))
errorpercentage = (error/X.shape[0]) * 100
print("\nError Percentage with SkLearn LDA: ",errorpercentage)
print("SkLearn LDA Slope: ",lda.coef_)
print("SkLearn LDA y_intercept: ",lda.intercept_)

#Calculation the Slope and Intercept of the Discriminant line produced by SKLearn's LDA
wLda = lda.coef_
yIntLda = lda.intercept_/wLda[0,1]
slopeLda = -wLda[0,0]/wLda[0,1]

# Plotting the Discriminant line produced using SKLearn's LDA
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = slopeLda * x_vals - yIntLda
plt.plot(x_vals, y_vals, 'y--', label = 'Discriminant Line with SKLearn\'s LDA') # Plotting the Discriminant Line

plt.legend()
plt.show()

'''
As we can see by the plot and the classification results: The discriminant line position and the classification results 
for both implementations are almost identical, minus small differences in precision.

We generate the slope and y_intercept of the discriminant line produced by manual Implementation using the formula provided.
For accessing the y_intercept of the discriminant line produced by SKLearn's LDA, we use the attribute 'intercept_'. 
But LDA does not directly provide the Intercept values. Since we are generating the slope values using the values of W. we need to 
get the values of W from SKLearn's LDA. The attribute 'coef_' directly provide the values of W. We need to calculate the intercept
value by dividing the value from 'intercept_' with the value of 2nd value of 'coef_'. Basically what we can understand from this is 
is that 'intercept_' gives the value of the threshold. This interpretation will give us the values of the intercept and slope to 
plot the discriminant line produced by SKLearn's LDA.  

The difference in the discriminant lines in the 2 methodologies is due to the values in W differing. Since we have kept the value of 
W in the manual implementation as 0 the classification error is shown as 31%. While the classification error of SKLearn's LDA is shown
as 16%. SKLearn by default sets the best possible value of threshold. To get the best classfication results in professor's methodology,
we need to set the threshold as -0.006. theis will make the discriminant lines of both the methodologies overlap. 
'''
