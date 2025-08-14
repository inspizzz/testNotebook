from datetime import datetime

class wiktors_class:
  def __init__(self, param1):
    self.param1 = param1
  def get_time(self):
    return datetime.now()
  def get_param(self):
    return self.param1
  def run(self, param):
    print(f"run function executed with parameter {param}")
    print(self.get_time())
    print(self.get_param())
