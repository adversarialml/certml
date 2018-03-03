"""Certify Machine Learning Pipeline"""

import numpy as np
import cvxpy as cvx
from certml.utils.data import get_projection_matrix
from certml.utils.cvx import cvx_dot


class UpperBound(object):

    def __init__(self, pipeline, norm_sq_constraint=None, max_iter=None, num_iter_to_throw_out=None,
                 learning_rate=None, verbose=True, print_interval=500):
        # Input Parameters
        self.norm_sq_constraint = norm_sq_constraint
        self.max_iter = max_iter
        self.num_iter_to_throw_out = num_iter_to_throw_out
        self.learning_rate = learning_rate

        # Debug Parameters
        self.verbose = verbose
        self.print_interval = print_interval

        self.pipeline = pipeline

        # Extract Pipeline Information
        cert_params = pipeline.cert_params()

        # Verify that the pipeline can be certified
        # This should really be returned to the user and not using asserts but whatever
        assert len(cert_params) == 2
        assert cert_params[0]['type'] == 'defense'
        assert cert_params[1]['type'] == 'classifier'

        defense = cert_params[0]
        classifier = cert_params[1]

        self.loss = classifier['loss']
        self.loss_grad = classifier['loss_grad']
        self.loss_cvx = classifier['loss_cvx']
        self.x = classifier['data']['features']
        self.y = classifier['data']['labels']

        start_w = classifier['params']['coef']
        self.init_w = np.zeros_like(start_w)
        self.init_b = 0

        self.constraints_cvx = defense['constraints_cvx']
        self.class_map = defense['data']['class_map']

        self.x_c = None
        self.y_c = None

        self.best_upper_bound = None
        self.best_upper_good_loss = None
        self.best_upper_bad_loss = None
        self.best_upper_params_norm_sq = None
        self.best_upper_good_acc = None
        self.best_upper_bad_acc = None

    def certify(self, epsilons):
        total_loss = np.zeros_like(epsilons)
        good_loss = np.zeros_like(epsilons)
        bad_loss = np.zeros_like(epsilons)

        for idx, epsilon in enumerate(epsilons):
            total_loss_i, good_loss_i, bad_loss_i = self.cert_rda(epsilon)

            total_loss[idx] = total_loss_i
            good_loss[idx] = good_loss_i
            bad_loss[idx] = bad_loss_i

        return total_loss, good_loss, bad_loss

    def cert_rda(self, epsilon):
        """Online Certification Algorithm using Regularized Dual Averaging"""

        # Initialize Variables as 0
        x_bs = np.zeros((self.max_iter, self.x.shape[1]))
        y_bs = np.zeros(self.max_iter)
        sum_w = np.zeros(self.x.shape[1])

        ##########
        # Line 1 #
        ##########

        # Initialize Sum of Gradients (z)
        sum_of_grads_w_sq = np.ones(self.x.shape[1])
        sum_of_grads_w = np.zeros(self.x.shape[1])
        sum_of_grads_b = 0

        # Initialize Upper Bound (U*)
        best_upper_bound = 10000

        # Initialize ??? (\lambda)
        current_lambda = 1 / self.learning_rate

        # Initialize Model (\theta)
        w = self.init_w
        b = self.init_b

        ##########
        # Line 2 #
        ##########

        for iter_idx in range(self.max_iter):

            #####################
            # Line 5 (1st Half) #
            #####################

            # Calculate gradient of loss
            grad_w, grad_b = self.loss_grad(self.x, self.y, w=w, b=b)

            if self.verbose:
                if iter_idx % self.print_interval == 0:
                    print("At iter %s:" % iter_idx)

            ##########
            # Line 3 #
            ##########

            # Find the attack point that maximizes loss function.
            # We do not know which class gives the maximum loss.
            # Pick the class with the worse (more negative) margin.
            worst_margin = None
            for y_b in set(self.y):

                class_idx = self.class_map[y_b]
                x_b = self.minimize_over_feasible_set(y_b, w)

                margin = y_b * (w.dot(x_b) + b)
                if (worst_margin is None) or (margin < worst_margin):
                    worst_margin = margin
                    worst_y_b = y_b
                    worst_x_b = x_b

            #####################
            # Line 5 (2nd Half) #
            #####################

            # Take the gradient with respect to that y
            if worst_margin < 1:
                grad_w -= epsilon * worst_y_b * worst_x_b
                grad_b -= epsilon * worst_y_b

            #####################
            # Line 4 (2nd Half) #
            #####################

            # Loss due to malicious data
            bad_loss = self.loss(worst_x_b, worst_y_b, w=w, b=b)

            # Store iterate to construct matching lower bound
            x_bs[iter_idx, :] = worst_x_b
            y_bs[iter_idx] = worst_y_b

            #####################
            # Line 4 (1st Half) #
            #####################

            # Loss due to clean data
            good_loss = self.loss(self.x, self.y, w=w, b=b)
            params_norm_sq = (np.linalg.norm(w) ** 2 + b ** 2)

            # Total Loss of the Poisoned Dataset
            total_loss = good_loss + epsilon * bad_loss

            if best_upper_bound > total_loss:
                self.best_upper_bound = total_loss
                self.best_upper_good_loss = good_loss
                self.best_upper_bad_loss = bad_loss
                self.best_upper_params_norm_sq = params_norm_sq
                self.best_upper_good_acc = np.mean((self.y * (self.x.dot(w) + b)) > 0)
                if worst_margin > 0:
                    self.best_upper_bad_acc = 1.0
                else:
                    self.best_upper_bad_acc = 0.0

            ##########
            # Line 6 #
            ##########

            # Update Gradient (z[t] = z[t-1] - g[t])
            sum_of_grads_w -= grad_w
            sum_of_grads_b -= grad_b

            # Update ?? (\lambda[t] = max(...))
            candidate_lambda = np.sqrt(np.linalg.norm(sum_of_grads_w) ** 2 + sum_of_grads_b ** 2) / np.sqrt(
                self.norm_sq_constraint)
            if candidate_lambda > current_lambda:
                current_lambda = candidate_lambda

                # Update Model (\theta[t] = z[t] / \lambda[t])
            w = sum_of_grads_w / current_lambda
            b = sum_of_grads_b / current_lambda

        # Save Candidate Attack Points
        self.x_c = x_bs
        self.y_c = y_bs

        return self.best_upper_bound, self.best_upper_good_acc, self.best_upper_bad_loss

    def minimize_over_feasible_set(self, y, w):
        y_ind = self.class_map[y]

        cvx_x = cvx.Variable(w.size)

        objective = self.loss_cvx(cvx_x, y=y_ind, w=w)
        constraints_all = self.constraints_cvx(cvx_x)

        constraints = list()
        constraints.append(constraints_all[y_ind])

        prob = cvx.Problem(objective, constraints)
        prob.solve(verbose=self.verbose)

        x_opt = np.array(cvx_x.value).reshape(-1)

        # TODO Project onto feasible set

        return x_opt
