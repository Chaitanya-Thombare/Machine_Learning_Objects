#Creating a Linear Regression's Object
class Linear_Regression:
  def __init__(self, learning_rate = 0.01, no_of_iterations = 1000):
    self.lr = learning_rate
    self.itrs = no_of_iterations
    self.weights = None
    self.bias = None

  #Fitting the training data (adjusting weights and bias)
  def fit(self, X, y):
    samples, features = X.shape
    cost_list = []
    self.weights = np.zeros(features)
    self.bias = 0

    #Actually adjusting weights and bias to reduce predicted and actual value difference
    for i in range(self.itrs):
      y_raw = np.dot(X, self.weights) + self.bias
      if(i != 0 and (i%1000) ==0):
        cost = -1 * (np.sum(y*np.log(y_raw) + (1-y)*np.log(1-y_raw))) / samples
        cost_list.append((i, cost))
        print(cost, i)
 
      dw = (2 / samples) * np.dot(X.T, (y_raw - y))
      db = (2 / samples) * np.sum(y_raw - y)
 
      self.weights -= ((self.lr) * dw)
      self.bias -= ((self.lr) * db)
      

    return cost_list

  #Using the adjusted weights to make predictions on test data
  #Using the adjusted weights to make predictions on test data
  def predict(self, X):
    predictions = np.dot(X, self.weights) + self.bias
    return predictions
