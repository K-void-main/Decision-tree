# %% Importing Libraries
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# %% Sample Data
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([0, 0, 0, 1, 1])

# %% Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X, y)

# %% Predictions
y_pred = model.predict(X)

# %% Confusion Matrix
cm = confusion_matrix(y, y_pred)
print(cm)
