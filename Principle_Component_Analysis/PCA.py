'''We will generate a random covariance matrix for each of our 2 classes. With this matrix we will generate a dataset for each of the class, with each class having 10
attributes and 1000 rows. We then combine the data and split it into Train and Test sets. We then calculate the Eigen values and Eigen vectors of the data. 
Then multiply it with the training and testing data and then calculate the classification error. We then try to reconstruct the data and calculate the Mean Square Error
and plot both of them in a graph.'''

# Importing Libraries
import numpy as np
from data_functions import combiningDataFor2Classes, generateCovMatrix, generateRandomData, setseed
from matplotlib import pyplot as plt
import sklearn.discriminant_analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if(__name__ == '__main__'):

    # Set seed value to get consistent random data
    setseed(42)

    # Set size
    size: int = int(1000)

    # Means for each of the 10 attributes of the 2 classes.
    mean1 = [4, 5, 4, 5, 4, 5, 4, 5, 4, 5]
    mean2 = [-4, -5, -4, -5, -4, -5, -4, -5, -4, -5]

    cov1 = generateCovMatrix(5, 40, 10)
    cov2 = generateCovMatrix(20, 20, 10)

    x1 = generateRandomData(mean1, cov1, size)
    x2 = generateRandomData(mean2, cov2, size)

    # Combining the Data and class values for the 2 classes
    X, Xc = combiningDataFor2Classes(size, x1, x2)

    # Splitting Data into Training and Testing Set with a 80:20 Split
    XTrain, XTest, XcTrain, XcTest = train_test_split(
        X, Xc, test_size=0.2, stratify=Xc)

    # Mean Centering the Training Data
    XTrainMC = X - np.mean(XTrain)

    # Calculating Principal Components
    # Calculating Eigen Values and Eigen Vectors
    D, E = np.linalg.eig(np.dot(XTrainMC.T, XTrainMC))
    sortedD = np.argsort(D)[::-1]  # Sorting Eigen Values in Decending order
    # Sorting Eigen Vectors on the bais or the sorted Eigen Values
    sortedE = E[:, sortedD]

    # Initializing MeanSquareError and Classification Error
    meanSquareError = np.zeros(5,)
    classificationError = np.zeros(6,)

    # Creating Object for LDA
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

    # This loop will first reduce dimentions of the sorted Eigen Vectors and then multiply it with the dataset to bring it into the reduced principal component space. But for the
    # first loop dimentions will not reduce as i is 0. After reducing the dimensions and bringing it into the principal component space, we will calculate the classification error
    # and then reconstruct the reduced data. We will then calculate the mean square error(MSE) of the reconstructed data. MSE will give you the variance between original data and
    # reconstructed data.
    for i in range(6):
        # Bring the training data into the principal component space while reducing dimensions
        YTrain = np.dot(XTrain, sortedE[:, 0:(10-i)])
        # Bring the testing data into the principal component space while reducing dimensions
        YTest = np.dot(XTest, sortedE[:, 0:(10-i)])
        eReduced = E[:, 0:(10-i)]  # Reducing dimensions of the Eigen Vectors
        lda.fit(YTrain, XcTrain)  # Fitting the LDA Model
        prediction = lda.predict(YTest)  # Performing Prediction on the model
        error = sum(abs(prediction - XcTest))  # Calculating number of errors
        # Calculating percentage of errors
        classificationError[i] = (error/YTest.shape[0]) * 100
        if(i > 0):
            XTrainRecon = np.dot(YTrain, eReduced.T)  # Reconstruction Data
            # Calculating Mean Square Error on Reconstructed Data
            meanSquareError[i-1] = mean_squared_error(XTrain, XTrainRecon)
    print(classificationError)
    print(meanSquareError)

    # Plotting the Mean Square error and the classification error
    plt.plot(range(1, 6), meanSquareError, label='Mean Square Error')
    plt.legend()
    plt.show()
    plt.plot(range(1, 7), classificationError, label='Classification Error(%)')
    plt.ylim([0, 100])
    plt.legend()
    plt.show()
