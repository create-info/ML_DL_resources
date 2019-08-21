from tensorflow.examples.tutorials.mnist import input_data 
import tensorflow as tf
import os


# 单个全连接神经网络
def dist_recognization():
  # # 获取真实的数据
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  # 1、准备数据，定义数字占位符：特征值[None,784],目标值[None,10]
  with tf.variable_scope("data"):
    x = tf.placeholder(tf.float32,[None,784])
    y_true = tf.placeholder(tf.int32, [None,10])
  # 2、建立模型，随机初始化权重w（784,10）偏置bias（10）
  with tf.variable_scope("model"):
    #  随机初始化权重和偏置
    weight = tf.Variable(tf.random_normal([784,10], mean=0.0, stddev=1.0), name="w")
    bias = tf.Variable(tf.constant(0.0, shape=[10]))
    # 预测None个样本的输出结果：matrix = [None,784]*[784,10]+[10] = [None,10]
    y_predict = tf.matmul(x, weight) + bias
  
  with tf.variable_scope("soft_cross"):
    #  3、计算损失loss,样本的平均交叉熵损失 
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
  
  # 4、梯度下降优化
  with tf.variable_scope("optimizer"):
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
  
  # 5、准确率计算
  with tf.variable_scope("accuracy"):
    equal_list = tf.equal(tf.arg_max(y_true,1),tf.arg_max(y_predict,1))
    accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))
    print(accuracy)
       
  # 使用tensorboard观察loss的变化情况
  # 收集变量
  tf.summary.scalar("losses", loss)
  tf.summary.scalar("acc",accuracy)

  # 高纬度变量收集
  tf.summary.histogram("weight", weight)
  tf.summary.histogram("bias", bias)


  # 定义一个初始化变量的op
  init_op = tf.global_variables_initializer()


  # 定义一个合并变量
  meraged = tf.summary.merge_all()

  # 开启会话，运行结果
  with tf.Session() as sess:
    # 初始化变量
    sess.run(init_op)

    # 建立events文件，然后写入
    filewriter = tf.summary.FileWriter("./test/",graph=sess.graph)


    # 迭代步数去训练，更新参数预测
    for i in range(2000):
      # 取出特征值和目标值
      minst_x, minst_y = mnist.train.next_batch(50)
      # 运行train_op,开始训练
      sess.run(train_op, feed_dict={x: minst_x, y_true: minst_y}) 

      # 写入每步训练的值
      summary = sess.run(meraged, feed_dict={x: minst_x, y_true: minst_y})
      filewriter.add_summary(summary, i)

      print("训练第%d步，准确率为：%f" % (i,sess.run(accuracy, feed_dict={x: minst_x, y_true: minst_y})))

  return None


if __name__ == "__main__":
  dist_recognization()  #手写数字识别
