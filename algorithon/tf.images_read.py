import tensorflow as tf
import os

def image_read(filelist):
  # 读取并转成张量
  # 1、构造文件队列
  file_queue = tf.train.string_input_producer(filelist)
  
  # 2、构造阅读器去读取文件的内容（默认读取一张图片）
  reader = tf.WholeFileReader()
  # 从队列中读取图片数据
  key,value = reader.read(file_queue)
  print(key)   # 文件名字的张量
  print(value)  # 文件内容的张量
  
  # 3.对读取的图片文件进行解码
  image = tf.image.decode_jpeg(value)
  print(image)

  # 4.处理图片的大小
  image_resize = tf.image.resize_images(image,[121,121])
  print(image_resize)

  # 注意：指定通道数以及对应的像素值类型
  image_resize.set_shape([121,121,3])

  # 5.进行批处理,批处理要求所有数据形状必须定义
  image_batch = tf.train.batch([image_resize],batch_size=4,num_threads=1,capacity=4)
  print(image_batch)

  return image_batch


if __name__ == "__main__":
  # 找到文件,得到文件名列表
  # file_name = os.listdir("./images/")
  # print(file_name)
  # 构建文件列表：路径+文件名
  # filelist = [os.path.join("./images/", file) for file in file_name]
  filelist = ['./images/a.jpg']
  print(filelist)

  image_resize = image_read(filelist)
  print(image_resize)

  # 开启会话，运行结果
  with tf.Session() as sess:
    # 定义一个线程协调器
    coordinator = tf.train.Coordinator()

    # 开启读文件的线程
    threads = tf.train.start_queue_runners(sess, coord=coordinator)
    
    # 打印读取的内容
    print(sess.run([image_resize]))

    # 读取结束后回收子线程
    coordinator.request_stop()
    coordinator.join(threads)
