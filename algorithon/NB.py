# 朴素贝叶斯分类
# 加载20类新闻分类数据并进行分割 > 生成文章特征词 > 使用NB setomator流程进行预估 
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def naviebayes():
  news = fetch_20newsgroups(subset='all')

  #进行数据分割
  x_train,x_test,y_train,y_test = train_test_split(news.data, news.target, test_size=0.25)
  
  #对数据集进行特征抽取
  tf = TfidfVectorizer()
  x_train = tf.fit_transform(x_train)
  print(tf.get_feature_names())

  # 以训练集中的词列表进行每篇文档的重要性统计
  x_test = tf.transform(x_test)
  
  #使用朴素贝叶斯算法进行统计
  mlt = MultinomialNB(alpha=1.0)
  print(x_train)
  # print(x_train.toarray())
  mlt.fit(x_train,y_train)

  y_predict = mlt.predict(x_test)
  print("预测的文章类别为：",y_predict)
  #得出准确率
  print("预测的准确率为：",mlt.score(x_test,y_test))
  #target_names是目标类别名称 
  print("每个类别精确率和召回率：",classification_report(y_test,y_predict,target_names=news.target_names))

  return None

if __name__ == "__main__":
  naviebayes()
