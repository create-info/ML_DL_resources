from google.colab import drive
drive.mount('/content/gdrive/')

# 指定当前的工作文件夹
import os
# 此处为google drive中的文件路径,drive为之前指定的工作根目录，要加上
os.chdir("gdrive/My Drive")

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# https://www.kaggle.com/psparks/instacart-market-basket-analysis#
# k-means对Instacart Market用户进行聚类
def kmeans():
  # 加载数据集
  prior = pd.read_csv("./data/order_products__prior.csv") # 购买记录
  orders = pd.read_csv("./data/orders.csv")  # 订单数据
  products = pd.read_csv("./data/products.csv") # 物品分区数据
  aisles = pd.read_csv("./data/aisles.csv")  # 物品数据

  # 合并四张表数据到一张表
  _mg = pd.merge(prior,products,on=['product_id','product_id'])
  _mg = pd.merge(_mg,orders,on=['order_id','order_id'])
  mt = pd.merge(_mg,aisles,on=['aisle_id','aisle_id'])
  print(mt.head(10))

  #使用交叉表进行分组
  cross = pd.crosstab(mt['user_id'],mt['aisle'])
  print(cross.head(10))

  #进行主成分分析
  pca = PCA(n_components=0.9)
  data = pca.fit_transform(cross)
  print(data.shape)

  #减少样本数量,只取前500个用户
  x = data[:500]
  print(x.shape)

  # 假设用户一共4个类别
  km = KMeans(n_clusters=4)
  km.fit(x)

  predict = km.predict(x)
  print(predict)

  # 显示聚类的结果
  plt.figure(figsize=(10,10))
  #每个类别给一个颜色
  colors = ['orange','green','blue','purple']
  col = [colors[i] for i in predict]
  plt.scatter(x[:,1],x[:,2],color=col) # 随便取两个特征画散点图
  plt.xlabel("1")
  plt.ylabel("2")
  plt.show()

  # 评估蕨类效果，计算所有样本的轮廓系数的平均值
  silhouette_score(x,predict)

  return None

if __name__ == '__main__':
  kmeans()
