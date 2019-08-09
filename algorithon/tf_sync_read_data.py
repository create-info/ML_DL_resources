import tensorflow as tf

# 使用tf读取数据，实现同步的模拟：先处理完数据，才能开始训练
# 1、首先定义队列
Queue = tf.FIFOQueue(3, tf.float32)  # 一个op

# 2、定义数据，并加入队列,[[005,0.01,0.02],]才是一个列表，否则[005,0.01,0.02]是一个张量
enq_many = Queue.enqueue_many(([0, 1, 2],))

# 3、从队列中取出数据，加1,再入队
out_data = Queue.dequeue()
data = out_data + 1
in_enque = Queue.enqueue(data)

# 开启会话去运行
with tf.Session() as sess:
  #初始化队列
  sess.run(enq_many)
  #处理数据
  for i in range(100):
    #tf中运行数据是有依赖性的，只要运行in_enque就可以重复执行步骤3
    sess.run(in_enque)

  #训练数据,这里假设是直接取出数据
  for i in range(Queue.size().eval()):
    print(sess.run(Queue.dequeue()))

