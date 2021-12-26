import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
from random import random

np.random.seed(10)
def makeRandomPoint(num, dim, upper):
    return np.random.normal(loc=upper, size=[num, dim])


# 产生一个k维随机向量，每维分量值在0~1之间
def random_point(k):
    return [random() for _ in range(k)]


# 产生n个k维随机向量
def random_points(k, n):
    return [random_point(k) for _ in range(n)]


# data
def create_logistic_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0,1,-1]])
    X, y = data[:,:2], data[:,-1]
    return train_test_split(X, y, test_size=0.3)

def create_svm_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    outputpath = 'data.csv'
    # outputpath是保存文件路径

    pd.set_option('display.max_columns', 5)  # a就是你要设置显示的最大列数参数
    pd.set_option('display.max_rows', 150)  # b就是你要设置显示的最大的行数参数
    pd.set_option('display.width', 100)  # x就是你要设置的显示的宽度，防止轻易换行

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df.to_csv(outputpath, sep=',', index=False, header=True)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    X, y = data[:,:2], data[:,-1]
    return train_test_split(X, y, test_size=0.3)