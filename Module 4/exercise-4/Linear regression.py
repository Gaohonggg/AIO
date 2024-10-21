import numpy as np


class CustomLinearRegression:
    def __init__(self,X_data, y_target, learning_rate=0.01, num_epochs=10000):
        self.num_samples = X_data.shape[0]
        self.X_data = np.c_[np.ones((self.num_samples,1)),X_data]
        self.y_target = y_target
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.theta = np.random.randn(self.X_data.shape[1],1)
        self.losses = []

    def compute_loss(self, y_hat, y_target):
        loss = (y_hat - y_target)**2
        return loss

    def predict(self, X_data):
        y_hat = np.dot( X_data, self.theta )
        return y_hat

    def fit(self):
        for epoch in range( self.num_epochs ):

            y_hat = self.predict( self.X_data )
            loss = self.compute_loss(y_hat,self.y_target)
            self.losses.append(loss)
            loss_grd = 2*(y_hat-self.y_target)/self.num_samples
            gradients = np.dot( self.X_data.T, loss_grd )
            self.theta = self.theta - self.learning_rate * gradients

            if epoch%50==0:
                print (f"Epoch: {epoch} - Loss: {loss}")

        return {
            "loss" : sum(self.losses)/len(self.losses),
            "weight" : self.theta
        }

def r2score(y_pred,y):
    rss = np.sum( (y-y_pred)**2 )
    tss = np.sum( (y - np.mean(y))**2 )
    r2 = 1 - (rss/tss)
    return r2

y_pred = np.array([1 , 2 , 3 , 4 , 5])
y = np.array([1 , 2 , 3 , 4 , 5])
print( r2score( y_pred , y ) )

y_pred = np.array([1 , 2 , 3 , 4 , 5])
y = np.array([3 , 5 , 5 , 2 , 4])
print( r2score( y_pred , y ) )
