import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X_data = [[1.4, 0.2],
          [1.3, 0.4],
          [4.0, 1.0],
          [4.7, 1.4]]
y_data = [0, 0, 1, 1]

X_data = np.array(X_data)
y_data = np.array(y_data)

x_test = np.array([2.4,0.8])

classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(X_data, y_data)
classifier.kneighbors(x_test)
classifier.predict(x_test)

