#Implementing Adams optimizer by combining momentum and rmsprop for more efficient learning
def compute_gradient_Adams(X, y_true, W, b, iteration, alpha=0.01, beta1=0.9, beta2=0.999, dW_rmsprop=np.zeros(len(W)), db_rmsprop=0, dW_momentum=np.zeros(len(W)), db_momentum=0):
  dW = np.dot(X.T, (predict(X, W, b) - y_true)) / len(y_true)
  db = np.sum(y_true - predict(X, W, b)) / len(y_true)

  #Calculating momentum
  dW_momentum = beta1 * dW_momentum + (1 - beta1) * dW
  db_momentum = beta1 * db_momentum + (1 - beta1) * db

  #Calculating rmsprop
  dW_rmsprop = beta2 * dW_rmsprop + (1 - beta2) * dW ** 2
  db_rmsprop = beta2 * db_rmsprop + (1 - beta2) * db ** 2

  #Bias correction
  dW_momentum = dW_momentum / (1 - beta1 ** iteration)
  db_momentum = db_momentum / (1 - beta1 ** iteration)
  dW_rmsprop = dW_rmsprop / (1 - beta2 ** iteration)
  db_rmsprop = db_rmsprop / (1 - beta2 ** iteration)

  #Combining rmsprop and momentum
  epsilon = 1e-8 #Very small value to prevent from dividing by 0

  dW_corrected = dW_momentum / np.sqrt(dW_rmsprop + epsilon)
  db_corrected = db_momentum / np.sqrt(db_rmsprop + epsilon)

  return dW_corrected, db_corrected
