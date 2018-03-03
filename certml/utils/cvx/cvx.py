"""CVX Utilities"""

import numpy as np
import cvxpy as cvx
from certml.utils.data import get_projection_matrix


def cvx_dot(a, b):
    """ CVX Dot Product

    Parameters
    ----------
    a : ???
    b : ???

    Returns
    -------
    dot : ???
        Dot product of a and b
    """
    return cvx.sum_entries(cvx.mul_elemwise(a, b))


class NearestPointFinder(object):
    """ Nearest Point Finder

    We can speed this up by expressing the constraints in one dimension,
    but let's see how it goes first.

    Solve the optimization problem:

    """

    def __init__(self, d):
        """ Nearest Point Finder

        Parameters
        ----------
        d : ???
            ???
        """

        self.cvx_c = cvx.Variable(1)
        self.cvx_y = cvx.Parameter(1)
        self.cvx_g = cvx.Parameter(d)
        self.cvx_theta = cvx.Parameter(d)
        self.cvx_centroid = cvx.Parameter(d)
        self.cvx_centroid_vec = cvx.Parameter(d)
        self.cvx_sphere_radius = cvx.Parameter(1)
        self.cvx_slab_radius = cvx.Parameter(1)

        # want grad of poisoned point = -g
        # grad of point (cyg, y) = -cg
        # so we want to find c > 0 such that n_poison/n * -cg = -g with 0 < n_poison as low as possible
        # and (cyg, y) in the feasible set.
        # since n_poison/n > 0, c must > 0. Then n_poison/n = 1/c,
        # so we want c to be as large as possible.
        # As a sanity check, we know that g itself is in conv(feasible set of gradients).
        # We can hope that either (-g, 1) or (g, -1) is in the feasible set of inputs,
        # because the gradients at each of those points is g
        # so if we let c = -1 then this should work.

        self.cvx_x = self.cvx_c * self.cvx_y * self.cvx_g
        self.cvx_x_c = self.cvx_x - self.cvx_centroid
        self.constraints = [
            cvx.norm(self.cvx_x_c, 2) < self.cvx_sphere_radius,
            cvx.abs(cvx_dot(self.cvx_centroid_vec, self.cvx_x_c)) < self.cvx_slab_radius,
            self.cvx_y * cvx_dot(self.cvx_theta, self.cvx_x) < 1,
            self.cvx_c > 0]

        self.objective = cvx.Maximize(self.cvx_c)

        self.prob = cvx.Problem(self.objective, self.constraints)

    def find_nearest_point(self, y, g, theta,
                           centroid, centroid_vec, sphere_radius, slab_radius,
                           verbose=False):
        """ Find Nearest Point

        Parameters
        ----------
        y : int
        g : ??
        theta :
        centroid : np.ndarray of shape (classes, dimensions)
        centroid_vec : np.ndarray of shape (dimensions,)
        sphere_radius : np.ndarray of shape (classes,)
        slab_radius : np.ndarray of shape (classes,)
        verbose : bool
            CVX verbose

        Returns
        -------
        c_opt :
            Optimal C value
        """
        self.cvx_y.value = y
        self.cvx_g.value = g.reshape(-1)
        self.cvx_theta.value = theta.reshape(-1)
        self.cvx_centroid.value = centroid.reshape(-1)
        self.cvx_centroid_vec.value = centroid_vec.reshape(-1)
        self.cvx_sphere_radius.value = sphere_radius
        self.cvx_slab_radius.value = slab_radius

        self.prob.solve(verbose=verbose)

        c_opt = self.cvx_c.value
        return c_opt


class Projector(object):
    """ Projector ???

    Solves the optimization problem:
    TODO
    """

    def __init__(self, use_sphere=True, use_slab=True):
        """ Projector ???

        Parameters
        ----------
        use_sphere
        use_slab
        """

        d = 3

        self.cvx_x = cvx.Variable(d)
        self.cvx_z = cvx.Parameter(d)
        self.cvx_centroid = cvx.Parameter(d)
        self.cvx_centroid_vec = cvx.Parameter(d)
        self.cvx_sphere_radius = cvx.Parameter(1)
        self.cvx_slab_radius = cvx.Parameter(1)

        self.cvx_x_c = self.cvx_x - self.cvx_centroid

        self.constraints = []
        if use_sphere:
            self.constraints.append(cvx.norm(self.cvx_x_c, 2) < self.cvx_sphere_radius)
        if use_slab:
            self.constraints.append(cvx.abs(cvx_dot(self.cvx_centroid_vec, self.cvx_x_c)) < self.cvx_slab_radius)

        self.objective = cvx.Minimize(cvx.norm(self.cvx_x - self.cvx_z, 2))

        self.prob = cvx.Problem(self.objective, self.constraints)

    def project_onto_feasible_set(self, z, centroid, centroid_vec, sphere_radius, slab_radius,
                                  verbose=False):
        """ Project onto Feasible Set

        Parameters
        ----------
        z : ??
        centroid : np.ndarray of shape (classes, dimension)
        centroid_vec : np.ndarray of shape (dimensions,)
        sphere_radius : np.ndarray of shape (classes,)
        slab_radius : np.ndarray of shape (classes,)
        verbose : bool

        Returns
        -------
        x_proj :
            Data projected onto feasible set
        """

        P = get_projection_matrix(z, centroid, centroid_vec)

        self.cvx_z.value = P.dot(z.reshape(-1))
        self.cvx_centroid.value = P.dot(centroid.reshape(-1))
        self.cvx_centroid_vec.value = P.dot(centroid_vec.reshape(-1))
        self.cvx_sphere_radius.value = sphere_radius
        self.cvx_slab_radius.value = slab_radius

        self.prob.solve(verbose=verbose)

        x_opt = np.array(self.cvx_x.value).reshape(-1)

        return x_opt.dot(P)
