#Predicting functions

#Initialize parameters
W = np.random.randn(len(features))
b = 0

#prediction function - predicts with current weight values
def predict(X, W, b):
  return np.dot(X, W) + b

#Cost computing function
def compute_cost(X, y_true, W, b):
  m = len(y_true)
  predictions = predict(X, W, b)
  cost = np.sum((predictions - y_true) ** 2) / (2 * m)
  return cost

#Training functions

def compute_gradient(X, y_true, W, b):
  dW = np.dot(X.T, (predict(X, W, b) - y_true)) / len(y_true)
  db = np.sum(y_true - predict(X, W, b)) / len(y_true)

  return dW, db

#Training function
def train(X_train, y_train, W, b, learning_rate, num_iterations):
  traincost_history = []
  valcost_history = []

  for i in range(1, num_iterations+1):
    dW, db = compute_gradient(X_train, y_train, W, b)
    W = W - (learning_rate * dW)
    b = b - (learning_rate * db)
    train_cost = compute_cost(X_train, y_train, W, b)


    if(len(traincost_history) != 0 and train_cost > traincost_history[-1]):
      print('Stopping...')
      break
    print(f"Iteration {i}: Training Cost = {train_cost}")
    print(f"Iteration {i}: Validation Cost = {compute_cost(X_test, y_test, W, b)}")

    traincost_history.append(train_cost)
    valcost_history.append(compute_cost(X_test, y_test, W, b))

  return W, b, traincost_history, valcost_history
