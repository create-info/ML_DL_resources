import tensorflow as tf
import os

def csv_read(filelist):
  """
    读取csv文件，
    ;filelist：文件路径+名字的列表
    ;return：读取的内容
  """
  # 1.构造文件队列
  file_queue = tf.train.string_input_producer(filelist)

  # 2.构造阅读器读取文件队列中的数据（一行一行读取）
  reader = tf.TextLineReader() #读取文本数据

  key, value = reader.read(file_queue)  #key是文件名，value是文件的内容（字节）
  print(key,value)

  # 3.对读取到的每行内容进行解码
  # 指定读取每个样本的每个列的类型，并指定默认值
  records = [["None"],["None"]] # 这里只有两个列

  example, label = tf.decode_csv(value, record_defaults=records)

  # 4.批处理读取多个样本,返回每一列,batch_size是一次取多少样本，capacity是队列中元素的最大数量
  example_batch, label_batch = tf.train.batch([example, label], batch_size=50, num_threads=1, capacity=145)
  print(example_batch, label_batch)

  return example_batch, label_batch
  # return example, label


if __name__ == "__main__":
  # 找到文件,得到文件名列表
  # file_name = os.listdir("./test1/")
  # print(file_name)
  # 构建文件列表：路径+文件名
  # filelist = [os.path.join("./test1/", file) for file in file_name]
  filelist = ['./test1/A.csv']
  print(filelist)

  example_batch, label_batch = csv_read(filelist)
  # print(example_batch)

  # 开启会话，运行结果
  with tf.Session() as sess:
    # 定义一个线程协调器
    coordinator = tf.train.Coordinator()

    # 开启读文件的线程
    threads = tf.train.start_queue_runners(sess, coord=coordinator)
    
    # 打印读取的内容
    # print(sess.run(example_batch))

    # 读取结束后回收子线程
    coordinator.request_stop()
    coordinator.join(threads)
