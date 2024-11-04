import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("creditcard.csv")
df = df.to_numpy()
X = df[:,:-1].astype(np.float64)
y = df[:,-1].astype(np.uint8)

X_b = np.concatenate( (np.ones((X.shape[0],1)),X), axis=1)

n_classes = np.unique(y,axis=0).shape[0]
n_samples = y.shape[0]

y_encoded = np.array(
    [ np.zeros(n_classes) for _ in range(n_samples) ]
)
for i in range( len(y_encoded) ):
    y_encoded[i,y[i]] = 1

val_size = 0.2
test_size = 0.125
randome_state = 2
X_train, X_val, y_train, y_val = train_test_split(X_b, y_encoded,
                                                  test_size=val_size,
                                                  random_state=randome_state,
                                                  shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=test_size,
                                                    random_state=randome_state,
                                                    shuffle=True)


standardscaler = StandardScaler()
X_train[:,1:] = standardscaler.fit_transform(X_train[:,1:])
X_val[:,1:] = standardscaler.transform(X_val[:,1:])
X_test[:,1:] = standardscaler.transform(X_test[:,1:])

def softmax(z):
    return np.exp(z) /( np.sum(np.exp(z),axis=1)[:,None] )

def predict(X, theta):
    z = np.dot(X, theta)
    y_hat = softmax(z)
    return y_hat

def compute_loss(y_hat,y):
    size = y.size
    return -np.sum( y*np.log(y_hat) )/size

def compute_gradient(X,y,y_hat):
    size = y.size
    return np.dot(X.T,y_hat-y)/size

def update_theta(theta,gradient,lr):
    return  theta - lr * gradient

def compute_accuracy(X,y,theta):
    y_hat = predict(X,theta)
    acc = ( np.argmax(y_hat,axis=1) == np.argmax(y,axis=1) ).mean()
    return acc

lr = 0.01
epochs = 30
batch_size = 1024
n_features = X_train.shape[1]

np.random.seed(randome_state)
theta = np.random.uniform(size=(n_features,n_classes))

train_accs = []
train_losses = []
val_accs = []
val_losses = []

for epoch in range(epochs):
    train_batch_accs = []
    train_batch_losses = []
    val_batch_accs = []
    val_batch_losses = []
    for i in range(0,X_train.shape[0],batch_size):
        X_i = X_train[i:i+batch_size]
        y_i = y_train[i:i+batch_size]

        y_hat = predict(X_i,theta)
        train_loss = compute_loss(y_hat,y_i)

        gradient = compute_gradient(X_i,y_i,y_hat)
        theta = update_theta(theta,gradient,lr)

        train_batch_losses.append(train_loss)

        train_acc = compute_accuracy(X_train,y_train,theta)
        train_batch_accs.append(train_acc)

        y_val_hat = predict(X_val,theta)
        val_loss = compute_loss(y_val_hat,y_val)
        val_batch_losses.append(val_loss)
        val_acc = compute_accuracy(X_val,y_val,theta)
        val_batch_accs.append(val_acc)

    train_batch_loss = sum(train_batch_losses)/len(train_batch_losses)
    val_batch_loss = sum(val_batch_losses) / len(val_batch_losses)
    train_batch_acc = sum(train_batch_accs) / len(train_batch_accs)
    val_batch_acc = sum(val_batch_accs) / len(val_batch_accs)

    train_losses.append(train_batch_loss)
    val_losses.append(val_batch_loss)
    train_accs.append(train_batch_acc)
    val_accs.append(val_batch_acc)
    print(f"\n EPOCH{epoch + 1}:\t Trainingloss: {train_batch_loss: .3f}\t Validationloss: {val_batch_loss: .3f}")

fig, ax = plt.subplots(2,2,figsize=(12,10))
ax[0 , 0].plot( train_losses )
ax[0 , 0].set( xlabel="Epoch", ylabel="Loss")
ax[0 , 0].set_title("Training Loss")

ax[0 , 1].plot( val_losses, "orange")
ax[0 , 1].set( xlabel="Epoch", ylabel="Loss")
ax[0 , 1].set_title("Validation Loss")

ax[1 , 0].plot( train_accs )
ax[1 , 0].set( xlabel ="Epoch", ylabel="Accuracy")
ax[1 , 0].set_title("Training Accuracy")

ax[1 , 1].plot( val_accs, "orange")
ax[1 , 1].set( xlabel ="Epoch", ylabel="Accuracy")
ax[1 , 1].set_title("Validation Accuracy")

plt.show()

val_set_acc = compute_accuracy(X_val,y_val, theta)
test_set_acc = compute_accuracy(X_test,y_test,theta)
print(val_set_acc)
print(test_set_acc)
