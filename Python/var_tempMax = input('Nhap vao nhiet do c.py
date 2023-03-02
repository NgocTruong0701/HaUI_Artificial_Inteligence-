import numpy as np
import pandas as pd
import csv

#var_tempMax = float(input('Nhap vao nhiet do cao nhat: '))
#var_tempMin = float(input('Nhap vao nhiet do thap nhat: '))
#var_wind = float(input('Nhap vao toc do gio: '))
#var_cloud = float(input('Nhap vao luong may: '))
#var_rel = float(input('Nhap vao do am: '))

#data = {
    #"Max Temperature" : [var_tempMax],
    #"Min Temperature" : [var_tempMin],
    #"Wind Speed" : [var_wind],
    #"Loud Cover" : [var_cloud],
    #"Relative Humidity" : [var_rel]
#}



def loadData():
    f = open("E:/Download/Nam_3_Ki_1/AI/Python/ThoiTiet_dulieu.csv", "r")
    data = csv.reader(f) #csv format
    data = np.array(list(data)) #convert thành ma trận
    data = np.delete(data, 0, 0) #xóa header
    data = np.delete(data, 0, 1) #xóa index
    f.close()
    trainSet = data[:] #bộ train 
    
    return trainSet

print(loadData())

def loadDataInput(path):
    f = open(path, "r")
    data = csv.reader(f) 
    data = np.array(list(data))
    data = np.delete(data, 0, 0) #xóa header
    f.close()
    return data[:1]


var_tempMax = input('Nhap vao nhiet do cao nhat: ')
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
df.to_csv("E:/Download/Nam_3_Ki_1/AI/Python/ThoiTiet_input.csv", index=False)
item = loadDataInput("E:/Download/Nam_3_Ki_1/AI/Python/ThoiTiet_input.csv")



