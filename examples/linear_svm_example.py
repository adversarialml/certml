"""Linear SVM Example"""

from certml.classifiers import LinearSVM
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

x, y = make_blobs(n_samples=100, n_features=2, centers=2)

model = LinearSVM(upper_params_norm_sq=1, use_bias=True)

model.fit(x, y)
pred = model.predict(x)

plt.subplot(1, 2, 1)
plt.title('True Labels')
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.subplot(1, 2, 2)
plt.title('Prediction')
plt.scatter(x[:, 0], x[:, 1], c=pred)
plt.show()
