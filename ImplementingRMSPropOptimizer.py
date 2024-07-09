#Implementing RMSprop for more efficient learning
def compute_gradient_rmsprop(X, y_true, W, b, alpha=0.01, beta=0.999, dW_rmsprop=np.zeros(len(W)), db_rmsprop=0):
  dW = np.dot(X.T, (predict(X, W, b) - y_true)) / len(y_true)
  db = np.sum(y_true - predict(X, W, b)) / len(y_true)

  dW_rmsprop = beta * dW_rmsprop + (1 - beta) * dW ** 2
  db_rmsprop = beta * db_rmsprop + (1 - beta) * db ** 2

  epsilon = 1e-8 #Very small value to prevent from dividing by 0

  #Dividing by the square root of the gradient reduces large steps in incorrect directions
  dW_corrected = dW / np.sqrt(dW_rmsprop + epsilon)
  db_corrected = db / np.sqrt(db_rmsprop + epsilon)

  return dW_corrected, db_corrected


