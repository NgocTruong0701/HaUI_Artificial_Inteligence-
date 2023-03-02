
import csv
import numpy as np
import math
from numpy import *
import pandas as pd

def loadData(): 
    f = open("E:/TriTueNhanTao/ThoiTiet_dulieu.csv")
    data = csv.reader(f) #csv format
    data = np.array(list(data))# convert to matrix
    data = np.delete(data, 0, 0)# delete header
    np.random.shuffle(data) # shuffle data
    f.close()
    trainSet = data[:362] #training data from 1->362
    testSet = data[362:]# the others is testing data
    return trainSet, testSet

def calcDistancs(pointA, pointB, numOfFeature=5): 
    tmp = 0
    for i in range(numOfFeature):
        tmp += (float(pointA[i]) - float(pointB[i])) ** 2
    return math.sqrt(tmp)

def kNearestNeighbor(trainSet, point, k): 
    distances = []
    for item in trainSet:
        distances.append({
            "label": item[-1],
            "value": calcDistancs(item, point)
        })
    distances.sort(key=lambda x: x["value"])
    labels = [item["label"] for item in distances]
    return labels[:k]

def findMostOccur(arr): 
    labels = set(arr) # set label
    ans = ""
    maxOccur = 0
    for label in labels:
        num = arr.count(label)
        if num > maxOccur:
            maxOccur = num  
            ans = label
    return ans

var_tempMax = input('Nhap nhiet do cao nhat: ')
var_tempMin = input('Nhap vao nhiet do thap nhat: ')
var_wind = input('Nhap vao toc do gio: ')
var_cloud = input('Nhap vao luong may: ')
var_rel = input('Nhap vao do am: ')

data = {
    "Max Temperature" : [var_tempMax],
    "Min Temperature" : [var_tempMin],
    "Wind Speed" : [var_wind],
    "Loud Cover" : [var_cloud],
    "Relative Humidity" : [var_rel]
}

df = pd.DataFrame(data)

df.to_csv("E:/TriTueNhanTao/ThoiTiet_test.csv", index=False)

def loadData2(): 
    f = open("E:/TriTueNhanTao/ThoiTiet_test.csv")
    data = csv.reader(f) #csv format
    data = np.array(list(data))# convert to matrix
    data = np.delete(data, 0, 0)# delete header
    f.close()
    trainSet1 = data[:1] 
    return trainSet1

if __name__ == "__main__":
    
    with open("E:/TriTueNhanTao/ThoiTiet_daxuly.csv",mode="w") as f:
        trainSet, testSet = loadData()
        numOfRightAnswer = 0
        writer = csv.writer(f)
        #way to write to csv file
        writer.writerow(['Label', 'Predicted'])
        for item in testSet:
            knn = kNearestNeighbor(trainSet, item, 5)
            answer = findMostOccur(knn)
            numOfRightAnswer += item[-1] == answer
            writer.writerow([item[-1], answer])
    print("\n\nAccuracy = ", numOfRightAnswer/len(testSet))
    f.close()
            
    with open("E:/TriTueNhanTao/ThoiTiet_testDL.csv",mode="w") as f:
        trainSet1 = loadData2()
        trainSet, testSet = loadData()
        writer = csv.writer(f)
        #way to write to csv file
        writer.writerow(['Label', 'Predicted'])
        for i in trainSet:
            knn1 = kNearestNeighbor(trainSet1, i, 5)
            answer1 = findMostOccur(knn1)
            writer.writerow([i[-1], answer1])
        print("\nLabel: {} -> predicted: {}".format(i[-1], answer1))
    f.close()
        





