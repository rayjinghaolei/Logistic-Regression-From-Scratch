import numpy as np

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

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()           
    weights = gradAscent(dataMat, labelMat)
    print(weights)
    #random shuffle
    #training set, test set 80/20
    #accuracy
    
