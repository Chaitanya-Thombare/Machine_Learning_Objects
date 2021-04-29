#Making parameters
def generic_parameters(layer_dims, depth):
  for i in range(1, depth):
    par['W'+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])*(0.001)
    par['b'+str(i)] = np.zeros((layer_dims[i], 1))

#Updating Parameters
def generic_update_parameters(y, m, lr, depth):
  k = 2/m
  der = {}
  der['dZ' + str(depth-1)] = (output['Z' + str(depth-1)] - y)*k
  for i in range(depth-1, 0, -1):
    der['dW'+str(i)] = np.dot(der['dZ'+str(i)], output['Z' + str(i-1)].T)*k
    der['db'+str(i)] = np.sum(der['dZ'+str(i)], axis=1, keepdims=True)*k
    der['dZ'+str(i-1)] = np.dot(par['W'+str(i)].T, der['dZ'+str(i)])*k
    par['W'+str(i)] -= (lr * der['dW'+str(i)])
    par['b'+str(i)] -= (lr * der['db'+str(i)])

#NN for classification.
class generic_neural_network_classification:
  def __init__(self, layer_dims, depth, learning_rate=0.01, no_of_iterations=1000):
    self.lr = learning_rate
    self.itrs = no_of_iterations
    generic_parameters(layer_dims, depth)

  #Fitting the training data (adjusting weights and bias)
  def fit(self, X, y, print_cost = 500):
    samples, features = X.shape
    output['Z0'] = X
    output['A0'] = np.zeros((X.shape))
    for i in range(self.itrs):
      if(i != 0 and (i % print_cost) == 0):
        cost =(- 1 / samples) * np.sum(y * np.log(output['A' + str(depth-1)]) + (1 - y) * (np.log(1 - output['A' + str(depth-1)])))
        print(i, cost)

      for i in range(1, depth):
        output['Z' + str(i)] = np.dot(par['W' + str(i)], output['Z' + str(i-1)]) + par['b' + str(i)]
        if i != (depth-1):
          output['A' + str(i)] = activation(output['Z' + str(i)], Tanh=True)
        else:
          output['A' + str(i)] = activation(output['Z' + str(i)], Sigmoid=True)

      generic_update_parameters(y, samples, self.lr, depth)

  def predict(self, X):
    result = {}
    result['Z0'] = X
    for i in range(1, depth):
      result['Z' + str(i)] = np.dot(par['W' + str(i)], result['Z' + str(i-1)]) + par['b' + str(i)]
      if i != (depth-1):
        result['A' + str(i)] = activation(result['Z' + str(i)], Tanh=True)
      else:
        result['A' + str(i)] = activation(result['Z' + str(i)], Sigmoid=True)
  
    return result['A' + str(depth-1)]

#NN for regression.
class generic_neural_network_regression_1:
def __init__(self, layer_dims, depth, learning_rate=0.01, no_of_iterations=1000):
  self.lr = learning_rate
  self.itrs = no_of_iterations
  generic_parameters(layer_dims, depth)

#Fitting the training data (adjusting weights and bias)
def fit(self, X, y, print_cost = 500):
  samples, features = X.shape
  output['Z0'] = X
  for i in range(self.itrs):
    if(i != 0 and (i % print_cost) == 0):
      cost = mean_absolute_error(y, output['Z' + str(depth-1)])
      print(i, cost)

    for j in range(1, depth-1):
      output['Z' + str(j)] = np.dot(par['W' + str(j)], output['Z' + str(j-1)]) + par['b' + str(j)]
      output['Z' + str(j)] = activation(output['Z' + str(j)], Tanh=True)
    output['Z' + str(depth-1)] = np.dot(par['W' + str(depth-1)], output['Z' + str(depth-2)]) + par['b' + str(depth-1)]

    generic_update_parameters(y, samples, self.lr, depth)

def predict(self, X):
  result = {}
  result['Z0'] = X
  for j in range(1, depth-1):
    result['Z' + str(j)] = np.dot(par['W' + str(j)], result['Z' + str(j-1)]) + par['b' + str(j)]
    result['Z' + str(j)] = activation(result['Z' + str(j)], Tanh=True)
  result['Z' + str(depth-1)] = np.dot(par['W' + str(depth-1)], result['Z' + str(depth-2)]) + par['b' + str(depth-1)]
  return result['Z' + str(depth-1)]
