# 使用决策树进行泰坦尼克号生存预测
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz

def decision():
  # 1.获取数据
  # titan = pd.read_csv("https://www.kaggle.com/c/titanic/download/train.csv")
  titan = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
  # 2.处理数据，找出特征值和目标值,这里只选取了3个特征，其中Pclass表示乘客的社会阶层
  x = titan[['pclass','age','sex']]
  y = titan['survived'] 
  print(x)
  # print(y)
  # age的缺失值处理
  x['age'].fillna(x['age'].mean(),inplace=True)

  # 分割数据到训练集和测试集
  x_train, x_test, y_train ,y_test = train_test_split(x,y,test_size = 0.25)

  # 特征工程，将年龄和性别进行one-hot编码
  dict = DictVectorizer(sparse=False) # 默认返回的是sparse矩阵的格式  
  x_train = dict.fit_transform(x_train.to_dict(orient="records")) #to_dict将数据转为字典 

  print(dict.get_feature_names())
  x_test = dict.fit_transform(x_test.to_dict(orient="records"))
  print(x_train)

  # 使用决策树算法,限制树的深度，默认使用的是基尼系数
  ctf = DecisionTreeClassifier(max_depth=5)
  ctf.fit(x_train, y_train)
  # 预测准确率
  print("预测的准确率为:",ctf.score(x_test,y_test))

  #导出决策树的结构
  export_graphviz(ctf,out_file="./tree.dot",feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male'])

if __name__ == "__main__":  
  decision()
