import csv
import numpy as np
import math
import pandas as pd


def loadData(path):
    f = open(path, "r")
    data = csv.reader(f) #csv format
    data = np.array(list(data)) #convert thành ma trận
    data = np.delete(data, 0, 0) #xóa header
    data = np.delete(data, 0, 1) #xóa index
    np.random.shuffle(data) # trộn bộ dữ liệu
    f.close()
    trainSet = data[:340] #bộ train từ 1 -> 340 
    testSet = data[340:] #bộ test là còn lại (427-340 = 87)
    return trainSet, testSet

def loadDataInput(path):
    f = open(path, "r")
    data = csv.reader(f) 
    data = np.array(list(data))
    data = np.delete(data, 0, 0) #xóa header
    f.close()
    return data[:1]

def calcDistancs(pointA, pointB, numOfFeature=5): #ham tinh khoang cach giua 2 diem
    tmp = 0
    for i in range(numOfFeature):
        tmp += math.pow(float(pointA[i]) - float(pointB[i]),2)
    return math.sqrt(tmp)

def getValue(x): #hỗ trợ cho hàm sắp xếp ở dưới
    return x["value"]

def kNearestNeighbor(trainSet, point, k): # hàm tìm k dữ liệu gần nhất
    distances = []
    for item in trainSet:
        distances.append({
            "label": item[-1], #nhãn
            "value": calcDistancs(item, point) #khoảng cách 2 điểm
        })

    distances.sort(key=getValue) # sắp xếp dựa trên value
    labels = [item["label"] for item in distances] #lấy các nhãn của bộ dữ liệu
    return labels[:k] #lấy k nhãn gần nhất

def findMostOccur(arr): #hàm tìm nhãn xuất hiện nhiều nhất (trạng thái thời tiết xuất hiện nhiều nhất)
    labels = set(arr) # set label
    ans = ""
    maxOccur = 0
    for label in labels:
        num = arr.count(label)
        if num > maxOccur: #nếu số lượng nhãn này lớn hơn số lượng nhãn lớn nhất trước đó tìm được thì gán lại
            maxOccur = num
            ans = label
    return ans # đưa ra nhãn xuất hiện nhiều nhất


if __name__ == "__main__":
    trainSet,testSet = loadData("E:/Download/Nam_3_Ki_1/AI/Python/ThoiTiet_dulieu.csv") #đọc đưa ra bộ train và test
    numOfRightAnwser = 0 #số lượng nhãn dự đoán đúng
    f = open("E:/Download/Nam_3_Ki_1/AI/Python/ThoiTiet_label_va_predicted.csv", "w") #tạo file để ghi hoặc tạo nó nếu chưa tồn tại
    writer = csv.writer(f) #ghi dạng csv
    for item in testSet: #chạy từng dữ liệu trong bộ test
        knn = kNearestNeighbor(trainSet, item, 5) #tính k hàng xóm của bộ train gần dữ liệu này nhất 
        answer = findMostOccur(knn) #đưa ra nhãn xuất hiện nhiều nhất
        numOfRightAnwser += item[-1] == answer 
        writer.writerow([item[-1], answer]) #ghi vào file: nhãn và dự đoán
    print("Accuracy", numOfRightAnwser/len(testSet))  #tính độ chính xác
    f.close()   

    var_tempMax = input('Nhap vao nhiet do cao nhat: ') #đoạn nhập values mà muốn dự đoán
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
    df.to_csv("E:/Download/Nam_3_Ki_1/AI/Python/ThoiTiet_input.csv", index=False)#tạo 1 file input 

    item = loadDataInput("E:/Download/Nam_3_Ki_1/AI/Python/ThoiTiet_input.csv")#đọc file input đấy
    for i in item:
        knn = kNearestNeighbor(trainSet, i, 5)
        answer = findMostOccur(knn)
        print("predicted: {}".format(answer)) #đưa ra dự đoán