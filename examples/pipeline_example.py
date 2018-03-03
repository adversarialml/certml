""" Pipeline Example """

from certml.pipeline import Pipeline
from certml.classifiers import LinearSVM
from certml.defenses import DataOracle
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

x, y = make_blobs(n_samples=100, n_features=2, centers=2)

steps = [
    ('Data Oracle', DataOracle(mode='sphere', radius=5)),
    ('Linear SVM', LinearSVM(upper_params_norm_sq=1, use_bias=True))
]

pipeline = Pipeline(steps)

pipeline.fit_trusted(x, y)
pipeline.fit(x, y)

pred = pipeline.predict(x)

cert_params = pipeline.cert_params()
print('Cert Params: {}'.format(cert_params))

plt.subplot(1, 2, 1)
plt.title('True Labels')
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.subplot(1, 2, 2)
plt.title('Prediction')
plt.scatter(x[:, 0], x[:, 1], c=pred)
plt.show()
