# ML_DL_resources
本文主要分享机器学习和深度学习相关书籍、项目、博客、学习路径、算法等，所记录的所有资料和链接均来自互联网，仅供个人学习参考使用。  
### 数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。
## 机器学习、深度学习框架
- [ml_dl_framework.doc](https://github.com/create-info/ML_DL_resources/blob/master/ml_dl_framework.doc)
## 机器学习、深度学习书单
- [ml_dl_books.doc](https://github.com/create-info/ML_DL_resources/blob/master/ml_dl_books.doc)
- [利用Python进行数据分析 第二版](https://github.com/BrambleXu/pydata-notebook)
## 机器学习、深度学习课程
- [ml_dl_course.doc](https://github.com/create-info/ML_DL_resources/blob/master/ml_dl_course.doc)
## 机器学习、深度学习数据集
- [ml_dl_datasets.doc](https://github.com/create-info/ML_DL_resources/blob/master/ml_dl_datasets.doc)
## 机器学习、深度学习学习路径
- [ml_dl_path.doc](https://github.com/create-info/ML_DL_resources/blob/master/ml_dl_path.doc)
- [AI算法工程师手册](http://huaxiaozhuan.com/)
## 机器学习、深度学习相关工具和类库
- [numpy官方文档](https://www.numpy.org/devdocs/reference/)
- [pands官方文档](http://pandas.pydata.org/)
- [matplotlib官方文档](https://matplotlib.org/)
- [python官方文档](https://docs.python.org/3.7/library/index.html)
- [scikit-learn官方文档](https://scikit-learn.org/stable/modules/classes.html)
- [xgboost安装文档](https://xgboost.readthedocs.io/en/latest/)    [xgboost github](https://github.com/dmlc/xgboost)
## 正则化技术
- [L1-Norm]
- [L2-Norm]  
- [Dropout]  
- [Max-Norm Regularization]
## 特征工程
特征工程本质是一项工程活动，目的是最大限度地从原始数据中提取特征以供算法和模型使用。  
- [1、特征的使用方案]  
特征的可用性评估,即哪些特征对我们模型有用,与业务高度相关,需要对业务有很深的理解;特征的覆盖率,准确率等。  
- [2、特征处理]  
主要数据清洗（清洗异常样本，噪声数据；采样：数据不均衡，样本权重）和预处理（单个特征、是否需要归一化，离散化；多个特征（是否需要组合特征，降维、特征选择等）；衍生变量（对原始数据加工、生成有商业意义的变量）。  
  [特征抽取和预处理](https://github.com/create-info/ML_DL_resources/blob/master/feature_extraction_preprocessing.ipynb)  
- [3、特征监控]  
特征有效性分析：特征的重要性，权重；特征监控：监控重要特征，防止特征质量下降，影响模型效果。
## 模型选择与验证
- 交叉验证  
将训练数据分成训练集以及验证集，然后取平均值作为最终的结果。  
- 网格搜索  
给定超参数的不同取值,依次对模型进行交叉验证来得到每个参数值下模型的效果。  
案例：[使用k-means预测用户的签到位置](https://github.com/create-info/ML_DL_resources/blob/master/algorithon/k-means-v2.py)
## 模型的效果评价
参考：sklearn.metrics.classification_report(y_test,y_predict,target_names=news.target_names)  
其中，target_names表示目标类别的名称
- 混淆矩阵
- 准确率
- 召回率
- F1-Score  
案例：[使用朴素贝叶斯进行新闻分类](https://github.com/create-info/ML_DL_resources/blob/master/algorithon/NB.py)
