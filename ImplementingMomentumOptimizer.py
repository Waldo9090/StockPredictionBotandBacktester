#Implementing gradient descent with momentum for more efficient learning
def compute_gradient_momentum(X, y_true, W, b, beta=0.9, dW_momentum=np.zeros(len(W)), db_momentum=0):
  dW = np.dot(X.T, (predict(X, W, b) - y_true)) / len(y_true)
  db = np.sum(y_true - predict(X, W, b)) / len(y_true)

  dW_momentum = beta * dW_momentum + (1 - beta) * dW
  db_momentum = beta * db_momentum + (1 - beta) * db

  return dW_momentum, db_momentum
