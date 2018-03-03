""" Pipeline Example """

from certml.pipeline import Pipeline
from certml.classifiers import LinearSVM
from certml.defenses import DataOracle
from certml.attacks.poison import LowerBound
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

x, y = make_blobs(n_samples=100, n_features=2, centers=2)

steps = [
    ('Data Oracle', DataOracle(mode='sphere', radius=5)),
    ('Linear SVM', LinearSVM(upper_params_norm_sq=1, use_bias=True))
]

# The pipeline that the attacker controls
pipeline_attacker = Pipeline(steps)
pipeline_attacker.fit_trusted(x, y)
pipeline_attacker.fit(x, y)

# The pipeline that the defender controls
pipeline_defender = Pipeline(steps)
pipeline_defender.fit_trusted(x, y)

# Get the attack points and poison the defenders training data
attack = LowerBound(pipeline=pipeline_attacker, norm_sq_constraint=1,
                    max_iter=1000, num_iter_to_throw_out=10,
                    learning_rate=1, verbose=False, print_interval=500)

x_c, y_c = attack.get_attack_points(0.3)

x_poisoned = np.append(x, x_c, axis=0)
y_poisoned = np.append(y, y_c, axis=0)

# Send the poisoned dataset to the defender
pipeline_defender.fit(x_poisoned, y_poisoned)
pred = pipeline_defender.predict(x)

plt.subplot(1, 2, 1)
plt.title('True Labels')
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.subplot(1, 2, 2)
plt.title('Prediction')
plt.scatter(x[:, 0], x[:, 1], c=pred)
plt.scatter(x_c[:, 0], x_c[:, 1], c=y_c, s=200, marker='*', edgecolors='r')
plt.show()
