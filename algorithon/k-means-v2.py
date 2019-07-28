
####    使用knn，预测用户的签到位置。
####    数据地址：https://www.kaggle.com/c/facebook-v-predicting-check-ins/data
####    时间复杂度o(n*k)：n为样本数量，k为单个样本特征的维度。如果不考虑特征维度的粒度为o(n)
####    空间复杂度o(n*k)：n为样本数量，k为单个样本特征的维度。如果不考虑特征维度的粒度为o(n)

from google.colab import drive
drive.mount('/content/gdrive/')

# 指定当前的工作文件夹
import os
# 此处为google drive中的文件路径,drive为之前指定的工作根目录，要加上
os.chdir("gdrive/My Drive")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd


def knncls():
  
  #读取数据
  data = pd.read_csv("./data/FBlocation/train.csv")
  print(data.head(10))
  
  #处理数据(因为预测的位置有很多，需要处理目标值)
  
  #1.缩小数据,查询数据刷选
  data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.7")
 
  #2.处理时间的数据，转为y-m-d h-m-s的格式
  time_value = pd.to_datetime(data['time'],unit='s')
  print(time_value)
  
  #3.把日期个数转换为字典格式
  time_value = pd.DatetimeIndex(time_value)
  
  #4.构造一些特征
  data['day'] = time_value.day
  data['hour'] = time_value.hour
  data['weekday'] = time_value.weekday
  
  #把时间戳特征(按列，sk-learn中0表示列)删除
  data.drop(['time'],axis=1)
  
  print(data)
  
  #把签到数量小于n个的目标位置删除
  place_count = data.groupby('place_id').count()
  tf = place_count[place_count.row_id > 3].reset_index()
  
  data = data[data['place_id'].isin(tf.place_id)]
  
  #取出数据当中的特征值和目标值
  y = data['place_id']
  x = data.drop(['place_id'],axis=1)
  x = data.drop(['row_id'],axis=1) 
  
  #进行数据的分割，划分为训练集和测试集
  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)
  
  #特征工程（标准化）经过处理后的数据均值为0，标准差为1:  标准话和drop row_id之前准确率只有4%，之后准确率提高到了0.659369527145359
  std = StandardScaler()
  #对训练集和测试集的特征进行标准化,fit就是进行一次平均值和标准差
  x_train = std.fit_transform(x_train) 
  x_test = std.transform(x_test)  # 前面一步已经计算好了标准差和均值
  
  
  # 进行算法流程，k是超参数
  knn = KNeighborsClassifier()
  
  # knn.fit(x_train,y_train)
  # y_predict = knn.predict(x_test)
  # print("预测目标的签到位置为;", y_predict)
  
  # #得出准确率
  # print("预测的准确率：", knn.score(x_test,y_test))
  
  # 进行网格搜索
  #构造参数的值
  param = {"n_neighbors": [3,5,10]}
  param = {"n_neighbors": [3,5,10]}
  # 每个n_neighbors的值都进行10折交叉验证
  gsc = GridSearchCV(knn,param_grid=param, cv=10)
  gsc.fit(x_train,y_train)

  #预测准确率
  print("在测试集合上的准确率:", gsc.score(x_test,y_test))
  print("在交叉验证中最好的结果：", gsc.best_score_)
  print("选择最好的模型是：", gsc.best_estimator_)
  print("每个超参数值每次交叉验证的结果：", gsc.cv_results_)

  return None
  
if __name__ == "__main__":
  knncls()
