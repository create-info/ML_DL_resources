import tensorflow as tf
import os

# 定义cifar的数据等命令行参数
FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('f', '', 'kernel')
# tf.app.flags.DEFINE_string("cifar_dir", "./data/cifar10/cifar-10-batches-bin/", "文件的目录")
# # 这里定义2个命令行参数
# tf.app.flags.DEFINE_string("cifar_tfrecords", "./tfrecords/cifar.tfrecords/", "存入tfrecords的文件")


class CifarRead(object):
  """
  读取二进制文件，写入tfrecords，然后读取tfrecords
  """
  def __init__(self, filelist):
    # 初始化文件列表
    self.filelist = filelist

    # 定义读取图片的一些属性
    self.height = 32
    self.weight = 32
    self.channel = 3
    # 二进制文件每张图片的字节
    self.label_bytes = 1
    self.image_bytes = self.height * self.weight * self.channel
    self.bytes = self.label_bytes + self.image_bytes


  def read_and_decode(self):
    # 1、构造文件队列
    file_queue = tf.train.string_input_producer(self.filelist)
  
    # 2、构造二进制文件读取器去读取文件的内容，每个样本的字节数
    reader = tf.FixedLengthRecordReader(self.bytes)
    # 从队列中读取二进制图片数据
    key, value = reader.read(file_queue)
    print("---------------图片文件名字的张量和内容的张量：key，value-----------------------")
    print(key)   # 文件名字的张量
    print(value)  # 文件内容的张量
    
    # 3.对读取的二进制文件进行解码
    label_image = tf.decode_raw(value, tf.uint8)
    print("---------------解码后的图片标签和内容：label_image-----------------------")
    print(label_image)

    # 4.分割出图片数据和标签数据
    label = tf.cast(tf.slice(label_image, [0], [self.label_bytes]), tf.int32)
    image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])

    # 5.对图片的特征数据进行形状的改变
    image_reshape = tf.reshape(image, [self.height, self.weight, self.channel])
    print("---------------图片的标签数据和图片的形状：label,image_reshape-----------------------")
    print(label,image_reshape)
    
    # 6.进行批处理,要求所有数据形状必须定义
    image_batch, label_batch = tf.train.batch([image_reshape, label],batch_size=10,num_threads=1,capacity=10)
    print("---------------批量读取文件：image_batch,label_batch-----------------------")
    print(image_batch,label_batch)

    return image_batch, label_batch

  #将图片内容与标签数据写入tfrecords文件 
  def write_2_tfrecords(self, image_batch, label_batch):
    """
    将图片中的特征值和目标值存入tfrecords文件
    image_batch:图片的特征值
    label_batch:图片的目标值
    """
    # 1、构造一个tfrecords文件
    writer = tf.python_io.TFRecordWriter(FLAGS.cifar_tfrecords)

    # 2、循环将所有样本写入文件，每张图片样本都要构造example协议块
    for i in range(10):
      # 取出第i张图片的特征值和目标值
      image = image_batch[i].eval().tostring()  #将张量转为字符串
      label = label_batch[i].eval()[0]
      # 构造一个样本的example协议块
      example = tf.train.Example(features=tf.train.Features(
          feature={ "image":tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),}))
      # 写入单独的样本
      writer.write(example.SerializeToString())
    # 关闭
    writer.close()
    return None
  #将tfrecords文件读取出来 
  def read_from_tfrecords(self):
    # 1.构造文件队列
    file_queue = tf.train.string_input_producer([FLAGS.cifar_tfrecords])
    print("-----------------------",file_queue)
    # 2.构造tfrecords文件阅读器读取文件example协议块，value是一个样本的example协议块
    reader = tf.TFRecordReader()
    key, value = reader.read(file_queue)

    # 3.解析example协议块,返回的是一个字典：样本的特征值和对应的标签
    features = tf.parse_single_example(value, features={
        "image":tf.FixedLenFeature([], tf.string),
        "label":tf.FixedLenFeature([], tf.int64),
    })
    print("---------------解析example协议块后的样本及其标签-----------------------")
    print(features["image"], features["label"])

    # 4.解码内容:若读取的内容格式是string则解码，int64和float32不用
    image =  tf.decode_raw(features["image"],tf.uint8)
    label = tf.cast(features["label"],tf.int32)

    # 固定图片的形状，方便批处理
    image_reshaped = tf.reshape(image, [self.height, self.weight, self.channel])

    print("--------------解码后的image以及label-----------------------")
    print(image_reshaped)
    print(label)

    # 批处理读取多张图片
    image_batch, label_batch = tf.train.batch([image_reshaped, label], batch_size=10, num_threads=1, capacity=10)

    return image_batch, label_batch


if __name__ == "__main__":
  # 找到文件,得到文件名列表
  file_name = os.listdir(FLAGS.cifar_dir)
  print(file_name)
  # 构建文件列表：路径+文件名
  filelist = [os.path.join(FLAGS.cifar_dir, file) for file in file_name if file[-3:] == "bin"]
  print(filelist)

  # 初始化类对象 
  cf = CifarRead(filelist)

  # image_batch, label_batch = cf.read_and_decode()

  image_batch, label_batch = cf.read_from_tfrecords()


  # 开启会话，运行结果
  with tf.Session() as sess:
    # 定义一个线程协调器
    coordinator = tf.train.Coordinator()

    # 开启读文件的线程
    threads = tf.train.start_queue_runners(sess, coord=coordinator)

    # 存进tfrecords文件，有eval(),所以必须得在session中运行
    # print("==============开始存储===============")
    # cf.write_2_tfrecords(image_batch, label_batch)
    # print("===============结束存储==============")

    # 打印读取的内容
    print("------------------打印读取的内容-------------------------")
    print(sess.run([image_batch, label_batch]))

    # 读取结束后回收子线程
    coordinator.request_stop()
    coordinator.join(threads)
