import numpy as np

def tanh(x):
   return np.tanh(x)

def del_tanh(x):
   val = 1 - np.square(np.tanh(x))
   return val

class NeuralNet:
   def __init__(self,guess,answer):
      self.X = guess
      self.weights1 = np.random.rand(self.X.shape[1],2)
      self.weights2 = np.random.rand(2,4)
      self.weights3 = np.random.rand(4,1)
      self.y = answer
      self.output = np.zeros(self.y.shape)

   def forward_pass(self):
      self.layer1 = tanh(np.dot(self.X, self.weights1))
      self.layer2 = tanh(np.dot(np.asarray(self.layer1), np.asarray(self.weights2)))
      self.output = tanh(np.dot(self.layer2, self.weights3))

   def backward_pass(self):
      del_weights3 = np.dot(self.layer2.T,2*(self.y - self.output)*del_tanh(self.output))
      del_weights2 = np.dot(self.layer1.T,(np.dot(self.weights3.T,2*(self.y - self.output)*del_tanh(self.output))*del_tanh(self.layer2)))
      del_weights1 = np.dot(self.X.T,(np.dot((np.dot(2*(self.y - self.output)*del_tanh(self.output),self.weights3.T)),self.weights2.T)*del_tanh(self.layer1)))

      self.weights1 += del_weights1
      self.weights2 += del_weights2
      self.weights3 += del_weights3

if __name__ == "__main__":
   X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
   y = np.array([[1],[-1],[1],[0]])
   nn = NeuralNet(X,y)

   for i in range(60):
      nn.forward_pass()
      nn.backward_pass()

print(nn.output) 
