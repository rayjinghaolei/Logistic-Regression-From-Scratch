import numpy as np
import random
import matplotlib.pyplot as plt
def loadDataSet():
    dataMat = []
    labelMat = []
    with open('iris.data','r') as reader:
        lines = reader.readlines()
        for line in lines:
            if line == '\n' :
                break
            line = line[:-1]
            line = line.split(',')
            lineData = []
            
            lineData.append(float(line[0]) * float(line[1]))
            lineData.append(float(line[2]) * float(line[3]))
            labelMat.append(1 if line[4] == 'Iris-setosa' else 0)
            dataMat.append(lineData)
        reader.close()
    return dataMat, labelMat
def splitDataSet(dataMat, labelMat):
    length = len(dataMat)
    data_and_label_Mat = list(zip(dataMat, labelMat))
    random.shuffle(data_and_label_Mat)
    test_length = length // 5
    training_length = length - test_length
    training_set = data_and_label_Mat[:training_length]
    testing_set = data_and_label_Mat[training_length:]
    training_data, training_label = zip(*training_set)
    testing_data, testing_label = zip(*testing_set)
    return (training_data,training_label, testing_data, testing_label)
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
        
    return weights.getA()

def predict(weight, testing_data, testing_label):
    weightMatrix = np.mat(weight)
    testing_data = np.mat(testing_data)
    result = np.dot(testing_data, weightMatrix)
    correct = 0
    total = len(testing_label)
    for i in range(len(result)):
        result[i] = 1 if result[i] >= 0 else 0
        if result[i] == testing_label[i]:
            correct += 1
    return correct / total;

def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()                                   
    dataArr = np.array(dataMat)                                         
    n = np.shape(dataMat)[0]                                            
    xcord1 = []; ycord1 = []                                            
    xcord2 = []; ycord2 = []                                            
    for i in range(n):                                                  
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,0]); ycord1.append(dataArr[i,1])    
        else:
            xcord2.append(dataArr[i,0]); ycord2.append(dataArr[i,1])    
    fig = plt.figure()
    ax = fig.add_subplot(111)                                           
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            
    x = np.arange(0, 30, 0.1)
    y = (-weights[0] - weights[0] * x) / weights[1]
    ax.plot(x, y)
    plt.title('BestFit')                                                
    plt.xlabel('X1'); plt.ylabel('X2')                                  
    plt.show()       

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    training_data,training_label, testing_data, testing_label = splitDataSet(dataMat, labelMat)
    weights = gradAscent(training_data, training_label)
    print(predict(weights, testing_data, testing_label))
    plotBestFit(weights)
    #random shuffle
    #training set, test set 80/20
    #accuracy
    
