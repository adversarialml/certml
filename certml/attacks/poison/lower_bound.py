"""Lower Bound Attack"""

from certml.certify.poison import UpperBound
import numpy as np


class LowerBound(object):

    def __init__(self, pipeline, norm_sq_constraint=None, max_iter=None, num_iter_to_throw_out=None,
                 learning_rate=None, verbose=True, print_interval=500):

        self.upper_bound = UpperBound(pipeline, norm_sq_constraint=norm_sq_constraint,
                                      max_iter=max_iter, num_iter_to_throw_out=num_iter_to_throw_out,
                                      learning_rate=learning_rate, verbose=verbose,
                                      print_interval=print_interval)

    def get_attack_points(self, epsilon=0.3):
        self.upper_bound.cert_rda(epsilon)
        x_c = self.upper_bound.x_c
        y_c = self.upper_bound.y_c

        num_iter = x_c.shape[0] - self.upper_bound.num_iter_to_throw_out

        idx_to_sample = np.random.choice(
            num_iter,
            size=int(np.round(epsilon * self.upper_bound.x.shape[0])),
            replace=False) + self.upper_bound.num_iter_to_throw_out

        return x_c[idx_to_sample, :], y_c[idx_to_sample]
