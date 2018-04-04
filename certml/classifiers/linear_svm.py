""" Linear SVM scikit-learn wrapper for rho squared parameter selection"""

import numpy as np
import cvxpy as cvx
import scipy.sparse as sparse
from sklearn import svm
from certml.certify import CertifiableMixin
from certml.utils.cvx import cvx_dot


class LinearSVM(svm.LinearSVC, CertifiableMixin):
    """Linear Support Vector Machine
    """
    def __init__(self, upper_params_norm_sq, use_bias, weight_decay=None):

        self._cert_x = None
        self._cert_y = None

        self.upper_params_norm_sq = upper_params_norm_sq
        self.rho_sq_tol = 0.01
        self.params_norm_sq = None

        if weight_decay is None:
            self.lower_wd_bound = 0.001
            self.upper_wd_bound = 256.0
        else:
            self.lower_wd_bound = 0.001
            self.upper_wd_bound = 2 * weight_decay - self.lower_wd_bound
            if self.upper_wd_bound < self.lower_wd_bound:
                self.upper_wd_bound = self.lower_wd_bound

        self.lower_weight_decay = self.lower_wd_bound
        self.upper_weight_decay = self.upper_wd_bound
        self.weight_decay = (self.upper_weight_decay + self.lower_weight_decay) / 2

        super(LinearSVM, self).__init__(tol=1e-6, loss='hinge',
                                        fit_intercept=use_bias, random_state=24,
                                        max_iter=100000, verbose=True)

    def fit(self, X, y, sample_weight=None):
        """ Fit the Linear Support Vector Machine

        Parameters
        ----------
        X : np.ndarray of shape (instances, dimensions)
            Input features
        y : np.ndarray of shape (instances,)
            Input labels
        sample_weight : np.ndarray of shape (instances,)
            ??? Weights for each training instance
        """
        assert np.all((y == 1) | (y == -1)), 'Input labels must be -1 or 1'

        self._cert_x = X
        self._cert_y = y

        # TODO This can get stuck in an infinite loop
        while (self.params_norm_sq is None) or \
                (self.upper_params_norm_sq > self.params_norm_sq) or \
                (np.abs(self.upper_params_norm_sq - self.params_norm_sq) > self.rho_sq_tol):

            print('Trying weight_decay %s..' % self.weight_decay)

            self.C = 1.0 / (X.shape[0] * self.weight_decay)
            super(LinearSVM, self).fit(X, y, sample_weight)

            params = np.reshape(self.coef_, -1)
            bias = self.intercept_[0]
            self.params_norm_sq = np.linalg.norm(params) ** 2 + bias ** 2

            if self.upper_params_norm_sq is None:
                break

            print('Current params norm sq = %s. Target = %s.' % (self.params_norm_sq, self.upper_params_norm_sq))
            # Current params are too small; need to make them bigger
            # So we should reduce weight_decay
            if self.upper_params_norm_sq > self.params_norm_sq:
                self.upper_weight_decay = self.weight_decay

                # And if we are too close to the lower bound, we give up
                if self.weight_decay < self.lower_wd_bound + 1e-5:
                    print('Too close to lower bound, breaking')
                    break

            # Current params are too big; need to make them smaller
            # So we should increase weight_decay
            else:
                self.lower_weight_decay = self.weight_decay

                # And if we are already too close to the upper bound, we should bump up the upper bound
                if self.weight_decay > self.upper_wd_bound - 1e-5:
                    self.upper_wd_bound *= 2
                    self.upper_weight_decay *= 2

            if (self.upper_params_norm_sq > self.params_norm_sq) or \
                    (np.abs(self.upper_params_norm_sq - self.params_norm_sq) > self.rho_sq_tol):
                self.weight_decay = (self.upper_weight_decay + self.lower_weight_decay) / 2

    def cert_params(self):
        """ Get the Certification Parameter Blob

        Returns
        -------
        params : dict
            Certification parameter blob
        """
        params = {
            'type': 'classifier',
            'loss': self._cert_loss,
            'loss_grad': self._cert_loss_grad,
            'loss_cvx': self._cert_loss_cvx,
            'data': {
                'features': self._cert_x,
                'labels': self._cert_y
            },
            'params': {
                'coef': self.coef_.flatten()
            }
        }
        return params

    def _cert_loss(self, X, Y, w=None, b=None, sample_weights=None):
        """ Calculate Hinge Loss

            Calculates the hinge loss.

            Parameters
            ----------
            X : np.ndarray of shape (instances, dimensions)
                Input Features
            Y : np.ndarray of shape (instances,)
                Input Labels
            sample_weights : None or np.ndarray of shape (???)
                ???

            Returns
            -------
            loss : float
                Hinge loss
            """
        if w is None or b is None:
            w = self.coef_.flatten()
            b = self.intercept_.flatten()

        if sample_weights is not None:
            sample_weights = sample_weights / np.sum(sample_weights)
            return np.sum(sample_weights * (np.maximum(1 - Y * (X.dot(w) + b), 0)))
        else:
            return np.mean(np.maximum(1 - Y * (X.dot(w) + b), 0))

    def _cert_loss_cvx(self, cvx_x, y=None, w=None, project=None):
        """ Hinge Loss CVX Optimization Problem

        Parameters
        ----------
        cvx_x
        y : int
            Data label
        w : np.ndarray of shape (dimensions,)
            Parameters
        project : np.ndarray of shape ()

        Returns
        -------
        loss_cvx
            CVX optimization objective
        """
        if project is not None:
            w = project.dot(w.reshape(-1))
        loss_cvx = cvx.Maximize(1 - y * cvx_dot(w.flatten(), cvx_x))
        return loss_cvx

    def _cert_loss_grad(self, X, Y, w=None, b=None):
        """ Gradient of Hinge Loss

        Parameters
        ----------
        w : np.ndarray of shape (dimensions,)
            Coefficients
        b : float
            Intercept
        X : np.ndarray of shape (instances, dimensions)
            Input Features
        Y : np.ndarray of shape (instances,)
            Input Labels

        Returns
        -------
        grad_w : np.ndarray of shape (dimensions,)
            Gradient of coefficients
        grad_b : float
            Gradient of intercept
        """
        if w is None or b is None:
            w = self.coef_.flatten()
            b = self.intercept_.flatten()

        margins = Y * (X.dot(w) + b).flatten()
        sv_indicators = margins < 1
        if sparse.issparse(X):
            grad_w = np.sum(
                -sparse.diags(np.reshape(Y[sv_indicators], (-1))).dot(
                    X[sv_indicators, :]), axis=0) / X.shape[0]
            grad_w = np.array(grad_w).reshape(-1)
        else:
            grad_w = np.sum(
                -np.reshape(Y[sv_indicators], (-1, 1)) * X[sv_indicators, :],
                axis=0) / X.shape[0]

        grad_b = np.sum(-np.reshape(Y[sv_indicators], (-1, 1))) / X.shape[0]

        return grad_w, grad_b
