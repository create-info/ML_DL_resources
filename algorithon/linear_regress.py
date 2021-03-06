import tensorflow as tf
import os
# 1、训练参数问题：trainable，学习率和步数的设置
# 2、添加权重参数，损失值等，在tensorboard中观察情况：第一，首先得收集loss以及weights
# 第二，合并变量写入事件文件

# 定义命令行参数,使用命令行参数启动:
# python dp_linearRegression.py --max_step=500 --model_dir="./lr_model"
# 1.首先定义有哪些参数需要在运行时指定
# 2.程序当中获取定义命令行参数

#第一个参数：名字，默认值，说明
tf.app.flags.DEFINE_integer("max_step",100,"模型训练的最大步数，默认是100")
tf.app.flags.DEFINE_string("model_dir","","模型的加载路径，不为空")

# 定义获取命令行参数名字的变量
FLAGES = tf.app.flags.FLAGS

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
  
  #1、收集tensor
  tf.summary.scalar("losses",loss)
  tf.summary.histogram("weights",weight)

  #2.定义合并tensor的op
  merged = tf.summary.merge_all()

  #定义一个初始化变量的op
  init_op = tf.global_variables_initializer()

  # 定义一个保存模型的实例
  saver = tf.train.Saver()

  #通过会话运行程序
  with tf.Session() as sess:
    # 初始化变量
    sess.run(init_op)

    # 打印随机最先初始化的权重和偏置，weight和偏置是op必须得run或者使用eval()获取值
    print("随机初始化的参数权重为: %f,偏置为: %f" % (weight.eval(), bias.eval())) 

    #建立事件文件
    fileWriter = tf.summary.FileWriter("./",graph=sess.graph)

# checkpoint
    # 加载模型，从之前训练的参数结果继续训练模型，将会覆盖模型中随机定义的参数。
    if os.path.exists("./checkpoint"):
      saver.restore(sess, FLAGES.model_dir)  # ???有问题

    # 循环去运行优化op
    for i in range(FLAGES.max_step):
      sess.run(train_op)

      # 运行merged的op
      summary = sess.run(merged) 
      fileWriter.add_summary(summary,i)

      print("第%d次优化后参数权重为: %f,偏置为: %f" % (i, weight.eval(), bias.eval()))
     
    saver.save(sess, FLAGES.model_dir)   # 指定模型保存的位置和名称

if __name__ == "__main__":
  myRegression()
