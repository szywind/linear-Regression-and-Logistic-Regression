import numpy as np
import numpy.linalg as nlg
import random

from math import *
def compGrad(u,v):
    gradu = exp(u) + v*exp(u*v) + 2*u -2*v-3
    gradv = 2*exp(2*v) + u*exp(u*v) -2*u + 4*v -2
    return (gradu, gradv)
 
def compHessian(u,v):
    graduu = exp(u) + v**2*exp(u*v) + 2
    gradvv = 4*exp(2*v) + u**2*exp(u*v) + 4
    graduv = exp(u*v) + u*v*exp(u*v) - 2
    return np.array([[graduu, graduv],[graduv, gradvv]])      

'''
use 1-d Taylor approximation
'''
def update1(times=5, yita=0.01):
    u = 0
    v = 0
    while(times):
        (du, dv) = compGrad(u,v)
        (u,v) = (u - yita*du, v - yita*dv)
        times -= 1
    return (u,v)
        
def E(u,v):
    return exp(u)+exp(2*v)+exp(u*v)+u**2-2*u*v+2*v**2-3*u-2*v


'''
use 2-d Taylor approximation
'''
def update2(times=5, yita=1):
    u = 0
    v = 0
    x = np.array([u,v])
    while(times):
        u = x[0]
        v = x[1]
        G = np.array(compGrad(u,v))
        H = compHessian(u,v)
        x  = x - yita*nlg.inv(H).dot(G)
        times -= 1
    return x


## Q13-Q15
def LinReg(trainSet):
    y = trainSet[:,-1]
    X = trainSet[:,:-1]
    return nlg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    
def genTrainSet(N=1000, D=2, noiseRatio=0.1):
    train = np.random.rand(N*D,1)*2-1
    noiseInd = np.random.choice(xrange(D*N), noiseRatio*N*D)
    train[noiseInd] = -train[noiseInd]
    train = train.reshape(N,D)
    label = np.sign(train[:,0]**2 + train[:,1]**2-0.6).reshape(-1,1)
    ##print train.shape
    ##print label.shape
    train = np.hstack((np.ones((N,1)), train,label))
    return train

def genTestSet():
    return genTrainSet()
     
def featTransform(dataSet):
    N = dataSet.shape[0]
    y = dataSet[:,-1].reshape(-1,1)
    X = dataSet[:,:-1]    
    X = np.hstack((X,(X[:,1]*X[:,2]).reshape(-1,1),(X[:,1]**2).reshape(-1,1),(X[:,2]**2).reshape(-1,1))) 
    # print X.shape 
    return np.hstack((X,y))      
             
def testLinReg(w, dataSet):
    N = dataSet.shape[0]
    y = dataSet[:,-1]
    X = dataSet[:,:-1]
    return np.sum(np.sign(X.dot(w))!=y)/(N+0.0)
    
def runLinReg(times, error, nonlinearFeat=False):
    Err = 0.0
    if nonlinearFeat:
        W = np.zeros((1,6))
    else:
        W = np.zeros((1,3))
    for i in xrange(times):
        # generate training set with noise
        trainSet = genTrainSet()
        testSet = genTestSet()
        if nonlinearFeat:
            trainSet = featTransform(trainSet)
            testSet = featTransform(testSet)
        
        # linear regression
        w = LinReg(trainSet)
        W += w
        
        # test and return in-sample error
        if error == "Ein":
            err = testLinReg(w, trainSet)
        elif error == "Eout":
            err = testLinReg(w, testSet)
        print "err = ", err
        Err += err
    return Err/times, W.mean(0)



## Q18-Q20
def readData(filename):
    with open(filename, "rt") as fl:
        data = []
        for line in fl:
            tmp = [float(elem) for elem in line.strip('\n').split()]
            data.append(tmp)  
        return np.array(data)

def sigmoid(s):
    return 1.0/(1+np.exp(-s))  
        
def LogReg(trainSet, yita, useSGD, T=2000):
    (N,D) = trainSet.shape
    D = D-1
    y = trainSet[:,-1].reshape(-1,1)
    X = trainSet[:,:-1]
    w = np.zeros((D,1))
    if useSGD:
        for i in xrange(T):
            ind = i%N
            xi = X[ind,:].reshape(1,-1)
            yi = y[ind]
            gradEin = -yi*xi.transpose().dot(sigmoid(-yi*np.dot(xi, w)))
            w = w - yita * gradEin        
    else:
        for i in xrange(T):
            # gradEin = - 1.0/N * ((y*np.ones((1,D))*X).transpose().dot(sigmoid(-y*np.dot(X, w))))
            gradEin =  -1.0/N * ((y*X).transpose().dot(sigmoid(-y*np.dot(X, w))))
            #print "gradEin = ", gradEin
            w = w - yita * gradEin
    return w
    
def testLogReg(w, dataSet): 
    N = dataSet.shape[0]
    y = dataSet[:,-1].reshape(-1,1)
    X = dataSet[:,:-1]
     
    return np.sum(np.sign(sigmoid(X.dot(w))-0.5)!=y)/(N+0.0)   #[wrong] np.sum(np.sign(sigmoid(X.dot(w)))!=y)  
        
def runLogReg(error, yita=0.001, useSGD=False):
    trainSet = readData("./ntumlone-hw3-hw3_train.dat")
    Ntr = trainSet.shape[0]
    trainSet = np.hstack((np.ones((Ntr,1)), trainSet))  
    testSet  = readData("././ntumlone-hw3-hw3_test.dat") 
    Nt = testSet.shape[0]
    testSet = np.hstack((np.ones((Nt,1)), testSet))
    
    # logistic regression
    w = LogReg(trainSet, yita, useSGD)
        
    # test and return in-sample error
    if error == "Ein":
        err = testLogReg(w, trainSet)
    elif error == "Eout":
        err = testLogReg(w, testSet)
    print "err = ", err
    return err