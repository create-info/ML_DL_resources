# 使用随机森林进行泰坦尼克号生存预测
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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

  rf = RandomForestClassifier()
  # 网格搜索与交叉验证
  param = {"n_estimators":[120,200,300,500,800,1200],"max_depth":[5,8,15,25,30]}
  gsv = GridSearchCV(rf,param_grid=param, cv=2)

  gsv.fit(x_train,y_train)
  print("预测的准确率为:",gsv.score(x_test,y_test))
  print("选择的参数为:",gsv.best_params_)

if __name__ == "__main__":  
  decision()
