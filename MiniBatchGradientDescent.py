#Implementing mini-batch gradient descent

def create_mini_batches(X, y, batch_size=128):
  Xmini_batches = []
  ymini_batches = []

  #Shuffle data while keeping X,y pairings
  indices = np.random.permutation(len(X))
  X_shuffled = X.iloc[indices]
  y_shuffled = y.iloc[indices]

  for i in range(0, len(X), batch_size):
    X_batch = X_shuffled.iloc[i:i+batch_size]
    y_batch = y_shuffled.iloc[i:i+batch_size]
    Xmini_batches.append(X_batch)
    ymini_batches.append(y_batch)

  return Xmini_batches, ymini_batches

#Training function for mini_batches
def train_batches(X_train, y_train, W, b, learning_rate, num_iterations):
  traincost_history = []
  valcost_history = []

  for i in range(1, num_iterations+1):
    Xmini_batches, ymini_batches = create_mini_batches(X_train, y_train)

    #Computing gradient for each mini batch
    for j in range(len(Xmini_batches)):
      dW, db = compute_gradient_Adams(Xmini_batches[j], ymini_batches[j], W, b, i)
      W = W - (learning_rate * dW)
      b = b - (learning_rate * db)

    train_cost = compute_cost(X_train, y_train, W, b)
    val_cost = compute_cost(X_test, y_test, W, b)

    if(len(traincost_history) != 0 and train_cost > traincost_history[-1]):
      print('Stopping...')
      break

    print(f"Iteration {i}: Training Cost = {train_cost}")
    print(f"Iteration {i}: Validation Cost = {val_cost}")

    traincost_history.append(train_cost)
    valcost_history.append(compute_cost(X_test, y_test, W, b))

  return W, b, traincost_history, valcost_history

