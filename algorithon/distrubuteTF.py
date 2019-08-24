
import tensorflow as tf
# 定义命令行参数
FLAGS = tf.app.flags.FLAGS
# 本地以worker启动，虚拟机中以ps启动 : python distrubuteTF.py --job_name="worker" --task_index=0
tf.app.flags.DEFINE_string("job_name", "", "启动ps还是worker")
tf.app.flags.DEFINE_integer("task_index", 0, "指定ps或worker中的哪一台服务器")

def main(argv):
 
  # 在使用钩子的时候需要定义一个全局参数:定义全局计算的op，给钩子列表当中的训练步数使用
  global_step = tf.contrib.framework.get_or_create_global_step()
 
  # 执行集群描述对象，哪些是参数服务器ps，哪些是worker服务器192.168.0.180(本机) 192.168.79.128(本地虚拟机)
  # 端口随便指定
  cluster = tf.train.ClusterSpec({"ps": ["192.168.79.128:2222"],
                  "worker": ["192.168.0.180:2223",]})
  # 创建不同的服务,job_name表示启动ps还是worker
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

  # 不同的服务器做不同的任务：ps去更新和保存参数，worker指定设备去运行模型计算
  if FLAGS.job_name == "ps":
    # 参数服务器啥也不用干，只需等待worker传递参数，
    server.join()
  else:
    worker_device = "/job:worker/task:0/cpu:0/"
    # 指定设备运行
    with tf.device(tf.train.replica_device_setter(
        worker_device = worker_device,
        cluster = cluster
    )):
      # 做一个矩阵乘法运算
      x = tf.Variable([[1,2,3,4]])   # 一行四列
      w = tf.Variable([[2],[3],[4],[5]])  # 四行一列
      y = tf.matmul(x,w)
    
    # 创建分布式会话
    with tf.train.MonitoredTrainingSession(
        master = "grpc://192.168.0.180:2223", # 指定主worker 
        is_chief = (FLAGS.task_index == 0), # 判断是否是主worker
        config = tf.ConfigProto(log_device_placement=True), # 打印设备信息
        hooks = [tf.train.StopAtStepHook(last_step=200)]  #运行200次
    ) as mon_sess:
       while not mon_sess.should_stop():  #没有异常停止
        mon_sess.run(y)
 

if __name__ == "__main__":
 tf.app.run()   #默认会调用main函数  
