#Creating a Logistic Regression's Object
class Logistic_Regression:
  def __init__(self, learning_rate = 0.01, no_of_iterations = 1000):
    self.lr = learning_rate
    self.itrs = no_of_iterations
    self.weights = None
    self.bias = None

  #Fittting the training data (adjusting weights and bias)
  def fit(self, X, y):
    samples, features = X.shape
    cost_list = []
    self.weights = np.zeros(features)
    self.bias = 0

    #Actually adjusting weights and bias to reduce predicted and actual value difference
    for i in range(self.itrs):
      y_raw = activation(np.dot(X, self.weights) + self.bias, Sigmoid = True)
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
  def classify(self, X):
    linear_probabilities = activation(np.dot(X, self.weights) + self.bias, Sigmoid = True)
    
    discrete_probabilities = np.zeros((X.shape[0], 1))
    for i in range(len(linear_probabilities)):
      if(linear_probabilities[i] > 0.5):
        discrete_probabilities[i][0] = 1

    return discrete_probabilities
