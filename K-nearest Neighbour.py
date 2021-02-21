#First KNN object
class K_Nearest_Neighbours_1:
  def __init__(self, k = 3):
    self.k = k

#Taking the inputs
  def plot(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train

#Actually running the KNN algorithm
  def classify(self, X_test):
    #Initializing required lists
    classes_and_distances = []#List to classes and distances in tuples -> [(class1, distance1), (class2, distance2)...]
    distances = []#List to store distances
    result = []#List to store results

    #Running KNN algoritm on each test input
    for x_test in X_test:
      for i in range(len(self.X_train)):
        classes_and_distances.append((y_train[i], distance(x_test, X_train[i])))

      classes_and_distances.sort(key = lambda x: x[1])
      classes_and_distances = classes_and_distances[ : self.k]

      classes_count = Counter(ele[0] for ele in classes_and_distances)
      result.append(classes_count.most_common()[0][0])#Using counter method and to find most common classes in nearest k neighbours
      
      #emptying the lists. Should have declared in this block.
      del classes_and_distances[:]
      del distances[:]

    return result

#Second KNN object
class K_Nearest_Neighbours_2:
  def __init__(self, k = 3):
    self.k = k

#Taking the inputs
  def plot(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train

#Actually running KNN
  def classify(self, X_test):
    result = np.array([])
    for x_test in X_test:
      distances = np.array([])#Array to store distances

      for i in range(len(self.X_train)):
        distances = np.append(distances, distance(x_test, X_train[i]))

      sorted_distances_index = np.argsort(distances)#Array to store indices of sorted distance array
      sorted_distances_index = sorted_distances_index[ : self.k]#Slicing array to choose nearest k neighbours

      
      nearest_neighbour_classes = [self.y_train[i] for i in sorted_distances_index]#List to store classes of nearest k neighbours
      classes_count = Counter(nearest_neighbour_classes)

      result = np.append(result, classes_count.most_common()[0][0])

    return result
