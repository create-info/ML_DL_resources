import tensorflow as tf

#使用tensorflow实现线性回归
def myRegression():
  # 使用线性回归预测
  # 创建变量作用域
  with tf.variable_scope("data"):
    #1、准备数据 特征值x,[100,1],目标值y,100
    x = tf.random_normal([100,1],mean=1.75,stddev=0.5,name="x_data")
    #矩阵相乘必须是二维的,这里假设知道w是0.7，b是0.8
    y_true = tf.matmul(x, [[0.7]]) + 0.8

  with tf.variable_scope("model"):
    #2、建立线性回归模型，一个特征，一个权重，一个偏置
    # 使用变量随机给定一个权重和一个偏置的值，然后使用梯度下降去计算损失，然后再去优化权重和偏置
    # trainable参数为false，该值不能跟着梯度下降一起优化，默认为True
    weight = tf.Variable(tf.random_normal([1,1], mean=0.0, stddev=1.0), name="w", trainable=True)
    bias = tf.Variable(0.0, name="b")

    y_predict= tf.matmul(x,weight) + bias
  
  with tf.variable_scope("loss"):
    #3 求均方误差损失函数 
    loss = tf.reduce_mean(tf.square(y_true-y_predict))

  with tf.variable_scope("optimizer"):
    #4 优化损失（梯度下降）,学习率为0.1，为0.3、0.25时梯度爆炸了。
    train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
  
  #定义一个初始化变量的op
  init_op = tf.global_variables_initializer()

  #通过会话运行程序
  with tf.Session() as sess:
    # 初始化变量
    sess.run(init_op)

    # 打印随机最先初始化的权重和偏置，weight和偏置是op必须得run或者使用eval()获取值
    print("随机初始化的参数权重为: %f,偏置为: %f" % (weight.eval(), bias.eval())) 

    #建立事件文件
    fileWriter = tf.summary.FileWriter("./content/",graph=sess.graph)


    # 循环去运行优化op
    for i in range(250):
      sess.run(train_op)
      print("第%d次优化后参数权重为: %f,偏置为: %f" % (i, weight.eval(), bias.eval()))
     

if __name__ == "__main__":
  myRegression()
