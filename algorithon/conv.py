from tensorflow.examples.tutorials.mnist import input_data 
import tensorflow as tf

def model():
  # 自定义卷积网络
  # 1.准备数据的占位符,其中x是[None,784],y_true是[None,true]
  with tf.variable_scope("data"):
    x = tf.placeholder(tf.float32, [None,784])
    y_true = tf.placeholder(tf.float32, [None,10])   # 输出是10个类别
  
  # 2.卷积层1；卷积（步长为1，5*5卷积核, 32个filter），激活(使用tf.nn.relu)和池化
  with tf.variable_scope("conv1"):
    
    w_conv1 = init_weight([5,5,1,32]) #32个5*5的卷积核，且输入通道为1，32为输出通道（即32个filter）
    b_conv1 = init_bais([32])   #有32个偏置
    
    # 对x形状进行改变，以作为tf的输入
    x_reshape = tf.reshape(x, [-1, 28, 28, 1]) # 1表示一个通道
    
    # padding上下左右一定都是1,[None, 28, 28, 1] -> [None, 28, 28, 32]
    x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)
    
    # 池化，步长strids=2, [None, 28, 28, 32] -> [None, 14, 14, 32]
    # ksize是池化窗口大小，为2*2，步长是2
    x_pool1 = tf.nn.max_pool(x_relu1, ksize = [1,2,2,1], strides=[1,2,2,1], padding="SAME")



  # 3.卷积层2；卷积（步长为1，5*5卷积核,64个filter），激活(使用tf.nn.relu)和池化
  with tf.variable_scope("conv2"):
    # 1. 初始化权重。[5,5,32,64], 偏置为64
    w_conv2 = init_weight([5,5,32,64]) #32个5*5的卷积核，且输入通道为1，64为输出通道
    b_conv2 = init_bais([64])   #有64个偏置
    # 2.卷积，激活，池化
    # [None, 14, 14, 32] -> [None, 14, 14, 64]
    x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, strides =[1,1,1,1], padding="SAME") + b_conv2)

    # 池化2*2, 步长为2, [None, 14, 14, 64] -> [None, 7, 7, 64]
    x_pool2 = tf.nn.max_pool(x_relu2, ksize = [1,2,2,1], strides=[1,2,2,1], padding="SAME")

  # 4.全连接层,[None, 7, 7, 64] -> [None, 7*7*64]*[7*7*64, 10] + [10] = [None, 10] 
  with tf.variable_scope("full_conn"):
      # 随机初始化权重和偏置
      w_fc = init_weight([7*7*64, 10])
      b_fc = init_bais([10])
      
      # 修改形状 [None, 7, 7, 64] -> [None, 7*7*64] 
      x_fc_reshape = tf.reshape(x_pool2, [-1, 7*7*64])
      
      # 执行矩阵预算，计算每个样本的10ge结果
      y_predict = tf.matmul(x_fc_reshape, w_fc) + b_fc


  return x, y_true, y_predict

# 定义初始化权重的函数
def init_weight(shape):
  w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
  return w

# 定义一个初始化偏置的函数
def init_bais(shape):
  b = tf.Variable(tf.constant(0.0,shape=shape))
  return b

# 卷积神经网络实现图片识别
def conv_nn():
  # 获取数据
  mnist = input_data.read_data_sets("./data/", one_hot=True)
  # 定义模型，得出输出
  x, y_true, y_predict = model()

  # 进行交叉熵损失计算
  # 1.求出所有样本的损失，然后求平均值，
  with tf.variable_scope("soft_cross"):
    # 求平均交叉熵损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict))
  # 梯度下降求出损失
  with tf.variable_scope("optimizer"):
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

  # 计算准确率
  with tf.variable_scope("acc"):
    equal_list = tf.equal(tf.argmax(y_true,1), tf.argmax(y_predict,1))
    # equal_list None个样本  [1,0,1,1,1,1,1,...,0.1]
    accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))

  # 定义一个初始化变量op
  init_op = tf.global_variables_initializer() 
  # 开启会话运行
  with tf.Session() as sess:
    sess.run(init_op)
    # 循环去训练
    for i in range(10000):
      # 取出真实存在的特征值和目标值
      mnist_x , mnist_y = mnist.train.next_batch(50)
      # 运行train_op进行训练
      sess.run(train_op, feed_dict={x:mnist_x,y_true:mnist_y})
      # 打印训练结果
      print("训练第%d步，准确率为：%f" % (i, sess.run(accuracy,feed_dict={x:mnist_x,y_true:mnist_y})) )

  return None

if __name__ == "__main__":
  conv_nn()  
