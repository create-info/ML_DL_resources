import tensorflow as tf
# 模拟异步子线程，存入样本，主线程读取样本

# 定义一个队列，里面含有1000个数据
Q = tf.FIFOQueue(1000, tf.float32)

# 定义要做的事情: 将var循环加1，放入队列中，
var = tf.Variable(0.0)

# 基于tf.assign_add()实现自增1
data = tf.assign_add(var, tf.constant(1.0))
en_q = Q.enqueue(data)

# 定义队列管理器，指定执行的子线程数:比如2，子线程该做什么事情:这里是操作队列Q执行入队操作
qr = tf.train.QueueRunner(Q,enqueue_ops=[en_q]*2) # 只要执行en_q，自动会执行其依赖data.

# 初始化变量op，里面的值是一个张量
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  #  初始化变量
  sess.run(init_op)

  # 开启线程管理器,用于管理线程，当主线程处理结束，关闭子线程
  coordinator = tf.train.Coordinator()

  # 真正开启子线程
  threads = qr.create_threads(sess, coord=coordinator, start=True)

  # 主线程不断去读取子线程写入队列中的数据
  for i in range(300):
    print(sess.run(Q.dequeue()))

  # 回收子线程:先询问，后终止
  coordinator.request_stop()
  coordinator.join(threads)
