import scipy.io as sio
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from sklearn import datasets

def dataset():
    mat=sio.loadmat('2d4c.mat',mdict=None,appendmat=True)

    data1=mat.get('a')
    data2=mat.get('moon')
    data3=mat.get('smile')
    data4=mat.get('b')
    label1=data1[:,2]
    label2=data2[:,2]
    label3=data3[:,2]
    label4=data4[:,2]
    data1=np.delete(data1, -1, axis=1)
    data2 = np.delete(data2, -1, axis=1)
    data3 = np.delete(data3, -1, axis=1)
    data4 = np.delete(data4, -1, axis=1)

    return data1,data2,data3,data4,label1,label2,label3,label4


def find_neighbor(j, x, eps):
    N = list()
    for i in range(x.shape[0]):
        temp = np.sqrt(np.sum(np.square(x[j] - x[i])))  # 计算欧式距离
        if temp <= eps:
            N.append(i)
    return set(N)


def DBSCAN(X, eps, min_Pts):
    k = -1
    neighbor_list = []  # 用来保存每个数据的邻域
    omega_list = []  # 核心对象集合
    gama = set([x for x in range(len(X))])  # 初始时将所有点标记为未访问
    cluster = [-1 for _ in range(len(X))]  # 聚类
    for i in range(len(X)):
        neighbor_list.append(find_neighbor(i, X, eps))
        if len(neighbor_list[-1]) >= min_Pts:
            omega_list.append(i)  # 将样本加入核心对象集合
    omega_list = set(omega_list)  # 转化为集合便于操作
    while len(omega_list) > 0:
        gama_old = copy.deepcopy(gama)
        j = random.choice(list(omega_list))  # 随机选取一个核心对象
        k = k + 1
        Q = list()
        Q.append(j)
        gama.remove(j)
        while len(Q) > 0:
            q = Q[0]
            Q.remove(q)
            if len(neighbor_list[q]) >= min_Pts:
                delta = neighbor_list[q] & gama
                deltalist = list(delta)
                for i in range(len(delta)):
                    Q.append(deltalist[i])
                    gama = gama - delta
        Ck = gama_old - gama
        Cklist = list(Ck)
        for i in range(len(Ck)):
            cluster[Cklist[i]] = k
        omega_list = omega_list - Ck
    return cluster

def run(X,eps,min_pts):
    eps = eps
    min_Pts = min_pts
    C = DBSCAN(X, eps, min_Pts)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=C)
    plt.show()

if __name__ == '__main__':
    d1,d2,d3,d4,l1,l2,l3,l4=dataset()
    X,_ = datasets.make_circles(n_samples=2000, factor=.6, noise=.02)   #生成回形数据
    run(X,0.1,2)
    run(d1,0.79,4)   #0.79
    run(d2,0.2,2)  #0.2
    run(d3,0.0859,2) #0.085
    run(d4,1,2)   #实在调不出来