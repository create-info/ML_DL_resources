# LRU最近最少使用算法

class LRUCache(object):
  def __init__(self, capacity):
    '''capcity:int'''
    self.cache = { }   # 存放缓存的key和value
    self.keys = []
    self.capacity = capacity
    
#     每次访问缓存中的key都要记录到list中，表示最近是访问了缓存中的哪个元素
  def visit_key(self , key):
      if key in self.keys:
        self.keys.remove(key)
      self.keys.append(key)
        
        
#     缓存满的情况下，删除最近最少使用的key和对应缓存中的值
  def elim_key(self):
      key = self.keys[0]
      self.keys = self.keys[1:]
      del self.cache[key]
      
 
  def get(self,key):
      if not key in self.cache:
        return -1
      self.visit_key(key)
      return self.cache[key]
    
    
  def put(self, key ,value):
      if not key in self.cache:
        if len(self.keys) == self.capacity:
#           清除最近最少使用的key
          self.elim_key()
#           加入缓存
        self.cache[key] = value
      self.visit_key(key)
    
def main():
      s = [["put","put","get","put","get","put","get","get","get"],[[1,1],[2,2],[1],[3,3],[2],[4,4],[1],[3],[4]]]
      obj = LRUCache(2)
      l = []
      for i,c in enumerate(s[0]):
        if(c == "get"):
          l.append(obj.get(s[1][i][0]))
        else:
          obj.put(s[1][i][0], s[1][i][1])
      print(l)
      
if __name__ == "__main__":
   main()  
    

# 最终输出[1, -1, -1, 3, 4]
