from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# dataset with only 2 useful features
X, y = make_classification(n_samples=200, 
                           n_features=2, 
                           n_informative=2, 
                           n_redundant=0, 
                           n_clusters_per_class=1, 
                           random_state=42)

# model
model = LogisticRegression().fit(X, y)

# mesh grid
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
                     np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# plot
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y, edgecolor="k")
plt.title("Logistic Regression Decision Boundary")
plt.show()
