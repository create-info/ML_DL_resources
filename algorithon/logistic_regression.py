# 逻辑回归预测恶性肿瘤
# 数据地址http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
# 总计699条数据，其中良性458（65.5%）恶性241（34.5%）
# 其中有16个缺失值，用？标记，最后一列是类别（2良性，4恶性）
# 第一列是id，第二列到第十列是医学特征
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def logistic_regression():
  #构建列标签名字
  column = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',
             'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin',
             'Normal Nucleoli','Mitoses','Class'
             ]
  # 读取数据并添加列名
  data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",names=column)
  # print(data)
  # 缺失值处理
  data = data.replace(to_replace='?',value=np.nan)
  data = data.dropna()  # 删除16行有缺失值的样本
  # print(data.shape)

  # 分割特征数据和标签数据并且切分训练集和测试集
  x_train,x_test,y_train,y_test = train_test_split(data[column[1:10]], data[column[10]], test_size=0.25)

  # print(x_train.shape)    #(512, 9)
  # print(x_test.shape)     #(171, 9)

  # 对特征值进行标准化处理
  sd = StandardScaler()
  x_train = sd.fit_transform(x_train)
  x_test = sd.fit_transform(x_test)

  # 进行逻辑回归预测,C是正则化力度
  lr = LogisticRegression(C=1.0)
  lr.fit(x_train,y_train)  # 基于对数似然损失
  y_predict = lr.predict(x_test)

  # 打印逻辑回归的系数
  print(lr.coef_)
  print("准确率：",lr.score(x_test,y_test))

  print("召回率:",classification_report(y_test,y_predict,labels=[2,4],target_names=["良性","恶性"]))
  

  return None
if __name__ == "__main__":
  logistic_regression()
