"""Data Utilities"""

import json
import numpy as np
import scipy.sparse as sparse
from certml.legacy import defenses
from certml.legacy import upper_bounds


# Migrated
class NumpyEncoder(json.JSONEncoder):
    """Numpy Encoder

    Encode numpy object as JSON.
    """
    def default(self, obj):
        """Override default encoder

        If the object is a 1D numpy array, encode it as JSON array.
        If the object is a numpy float, encode it as a JSON number.

        Parameters
        ----------
        obj : nd.array, np.floating, or other type
            Object to encode

        Returns
        -------
        json : str
            Encoded JSON blob
        """
        if isinstance(obj, np.ndarray):
            assert len(np.shape(obj)) == 1  # Can only handle 1D ndarrays
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(NumpyEncoder, self).default(obj)


def get_class_map():
    """ Get Mapping Between Class Labels and Indices

    The labels are stores as either -1 or 0, but we sometimes need them as 0
    or 1 for indexing into matrices. This "class map" makes it easy to convert
    as you can run class_map(Y).

    Returns
    -------
    class_map : dict
        Class to index mapping
    """
    return {-1: 0, 1: 1}  # Store class -1 as index 0 and 1 and index 1


# Migrated
def get_centroids(X, Y, class_map):
    """ Get Centroids for Each Class

    Parameters
    ----------
    X : np.ndarray of shape (instances, dimensions)
        Input features
    Y : np.ndarray of shape (instances,)
        Input labels
    class_map : dict
        Class to index mapping

    Returns
    -------
    centroids : np.ndarray of shape (classes, dimensions)
        Centroids of each class
    """
    num_classes = len(set(Y))
    num_features = X.shape[1]
    centroids = np.zeros((num_classes, num_features))
    for y in set(Y):            
        centroids[class_map[y], :] = np.mean(X[Y == y, :], axis=0)
    return centroids


# Migrated
def get_centroid_vec(centroids):
    """ Get Centroids Vector

    Get the unit vector that starts from the centroid of
    class 1 and points toward the centroid of class -1.

    Parameters
    ----------
    centroids : np.ndarray of shape (classes, dimensions)
        Centroids of each class

    Returns
    -------
    centroids_vec : np.ndarray of shape (1, dimensions)
        Unit vector from centroid of class 1 to centroid of class -1
    """
    assert centroids.shape[0] == 2
    centroid_vec = centroids[0, :] - centroids[1, :]
    centroid_vec /= np.linalg.norm(centroid_vec)
    centroid_vec = np.reshape(centroid_vec, (1, -1))
    return centroid_vec


# Migrated
def get_sqrt_inv_cov(X, Y, class_map):
    """ Get the Square Root of the Inverse Covariance Matrix for Each Class

    For each class, calculate the square root of the inverse covariance matrix
    of the features by singular value decomposition.

    Note: Can speed this up if necessary

    Parameters
    ----------
    X : np.ndarray of shape (instances, dimensions)
        Input features
    Y : np.ndarray of shape (instances)
        Input labels
    class_map : dict
        Class to index mapping

    Returns
    -------
    sqrt_inv_covs : np.ndarray of shape (classes, instances, dimensions)
        Square root of the inverse covariance matrix for each class

    """
    num_classes = len(set(Y))
    num_features = X.shape[1]
    sqrt_inv_covs = np.zeros((num_classes, num_features, num_features))

    for y in set(Y):            
        cov = np.cov(X[Y == y, :], rowvar=False)
        U_cov, S_cov, _ = np.linalg.svd(cov)
        sqrt_inv_covs[class_map[y], ...] = U_cov.dot(np.diag(1 / np.sqrt(S_cov)).dot(U_cov.T))

    return sqrt_inv_covs


# Migrated
def get_data_params(X, Y, percentile):
    """Helper Function to Calculate Useful Properties About the Dataset

    Note: Can speed this up if necessary

    Parameters
    ----------
    X : np.ndarray of shape (instances, dimensions)
        Input features
    Y : np.ndarray of shape (instances,)
        Input labels
    percentile : float
        Percentage of data to keep when setting radii

    Returns
    -------
    class_map : dict
        Class to index mapping
    centroids : np.ndarray of shape (classes, dimensions)
        Centroids of each class
    centroid_vec : np.ndarray of shape (dimensions,)
        Unit vector from centroid of class 1 to centroid of class -1
    sphere_radii : float
        Radius of sphere defense so the percentile criteria is met
    slab_radii : float
        Radius of slab defense so the percentile criteria is met
    """
    num_classes = len(set(Y))
    num_features = X.shape[1]
    centroids = np.zeros((num_classes, num_features))
    class_map = get_class_map()
    centroids = get_centroids(X, Y, class_map)

    # Get radii for sphere
    sphere_radii = np.zeros(2)
    dists = defenses.compute_dists_under_Q(    
        X, Y,
        Q=None,
        centroids=centroids,
        class_map=class_map,    
        norm=2)
    for y in set(Y):            
        sphere_radii[class_map[y]] = np.percentile(dists[Y == y], percentile)

    # Get vector between centroids
    centroid_vec = get_centroid_vec(centroids)

    # Get radii for slab
    slab_radii = np.zeros(2)
    for y in set(Y):            
        dists = np.abs( 
            (X[Y == y, :].dot(centroid_vec.T) - centroids[class_map[y], :].dot(centroid_vec.T)))            
        slab_radii[class_map[y]] = np.percentile(dists, percentile)

    return class_map, centroids, centroid_vec, sphere_radii, slab_radii


# Migrated
def add_points(x, y, X, Y, num_copies=1):
    """Add points to the dataset

    Parameters
    ----------
    x : np.ndarray of shape (instances, dimensions)
        Features of points to add
    y : np.ndarray of shape (instances,)
        Labels of points to add
    X : np.ndarray of shape (instances, dimensions)
        Dataset features
    Y : np.ndarray of shape (instances)
        Dataset labels
    num_copies : int
        Number of copies of the points to add

    Returns
    -------
    X_modified : np.ndarray of shape (instances, dimensions)
        Modified dataset features
    Y_modified : np.ndarray of shape (instances,)
        Modified dataset labels
    """
    if num_copies == 0:
        return X, Y

    if sparse.issparse(X):
        X_modified = sparse.vstack((
            X, 
            sparse.csr_matrix(
                np.tile(x, num_copies).reshape(-1, len(x)))))
    else:
        X_modified = np.append(
            X, 
            np.tile(x, num_copies).reshape(-1, len(x)), 
            axis=0)
    Y_modified = np.append(Y, np.tile(y, num_copies))
    return X_modified, Y_modified


# Migrated
def copy_random_points(X, Y, mask_to_choose_from=None, target_class=1, num_copies=1, 
                       random_seed=18, replace=False):
    """ Copy Random Points from a Dataset

    Parameters
    ----------
    X : np.ndarray of shape (instances, dimensions)
        Input features
    Y : np.ndarray of shape (instances,)
        Input labels
    mask_to_choose_from : np.ndarray of shape (instances,) and dtype=bool
        Only copy from points where mask_to_choose_from == True
    target_class : int
        Only copy from class target_class
    num_copies: int
        Number of points to copy
    random_seed : int
        Numpy random seed
    replace : bool
        Copy with replacement

    Returns
    -------
    X_modified : np.ndarray of shape (instances, dimensions)
        Copied dataset features
    Y_modified : np.ndarray of shape (instances,)
        Copied dataset labels
    """

    np.random.seed(random_seed)    
    combined_mask = (np.array(Y, dtype=int) == target_class)
    if mask_to_choose_from is not None:
        combined_mask = combined_mask & mask_to_choose_from

    idx_to_copy = np.random.choice(
        np.where(combined_mask)[0],
        size=num_copies,
        replace=replace)

    if sparse.issparse(X):
        X_modified = sparse.vstack((X, X[idx_to_copy, :]))
    else:
        X_modified = np.append(X, X[idx_to_copy, :], axis=0)
    Y_modified = np.append(Y, Y[idx_to_copy])
    return X_modified, Y_modified
    

# Migrated
def threshold(X):
    """ Set all Negative Values in X to 0

    Parameters
    ----------
    X : np.ndarray of shape (instances, dimensions)
        Input features

    Returns
    -------
    X_clip : np.ndarray of shape (instances, dimensions)
        Clipped input features
    """
    return np.clip(X, 0, np.max(X))


# Migrated
def rround(X, random_seed=3):
    """ Random Round

    Randomly round number according to the fractional component.
    E.g. The number 5.25 has a probability of 0.75 of being rounded
    to 5 and a probability of 0.25 of being rounded to 6.

    Parameters
    ----------
    X : np.ndarray of shape (instances, dimensions)
        Input features
    random_seed : int
        Numpy random seed

    Returns
    -------
    X_rround : np.ndarray of shape (instances, dimensions)
    """
    X_frac, X_int = np.modf(X)
    X = X_int + (np.random.random_sample(X.shape) < X_frac)
    return X


# Migrated
def project_onto_sphere(X, Y, radii, centroids, class_map):
    """ Project Dataset onto Sphere

    For each class, project the data onto a sphere with a
    particular radius centered at the class centroid.

    Note: If the data is already within the sphere, it is unchanged.

    Parameters
    ----------
    X : np.ndarray of shape (instances, dimensions)
        Input features
    Y : np.ndarray of shape (instances,)
        Input labels
    radii : np.ndarray of shape (classes,)
        Sphere radius for each class
    centroids : np.ndarray of shape (classes, dimensions)
        Centroid of each class
    class_map : dict
        Class to index mapping

    Returns
    -------
    X_proj : np.ndarray of shape (instances, dimensions)
        Projected features onto sphere
    """
    for y in set(Y):
        idx = class_map[y]        
        radius = radii[idx]
        centroid = centroids[idx, :]

        shifts_from_center = X[Y == y, :] - centroid
        dists_from_center = np.linalg.norm(shifts_from_center, axis=1)

        shifts_from_center[dists_from_center > radius, :] *= radius / np.reshape(dists_from_center[dists_from_center > radius], (-1, 1))
        X[Y == y, :] = shifts_from_center + centroid

        print("Number of (%s) points projected onto sphere: %s" % (y, np.sum(dists_from_center > radius)))

    return X
 

# Migrated
def project_onto_slab(X, Y, v, radii, centroids, class_map):
    """Project Dataset onto Slab

    For each class, project the data a slab given by the equation

    .. math
        |<X - Centroid1, Centroid1 - Cendroid2>| <= Radius

    Note: If the data is already in the slab, it is unchanged.

    Parameters
    ----------
    X : np.ndarray of shape (instances, dimensions)
        Input features
    Y : np.ndarray of shape (instances,)
        Input labels
    v : np.ndarray of shape (1, dimensions)
        Centroid vector (v^T x needs to be within radius of v^T centroid)
    radii : np.ndarray of shape (classes,)
        Slab radius for each class
    centroids : np.ndarray of shape (classes, dimensions)
        Centroid of each class
    class_map : dict
        Class to index mapping

    Returns
    -------
    X_proj : np.ndarray of shape (instances, dimensions)
        Projected features onto slab
    """
    v = np.reshape(v / np.linalg.norm(v), (1, -1))

    for y in set(Y):
        idx = class_map[y]
        radius = radii[idx]
        centroid = centroids[idx, :]

        # If v^T x is too large, then dists_along_v is positive
        # If it's too small, then dists_along_v is negative
        dists_along_v = (X[Y == y, :] - centroid).dot(v.T)
        shifts_along_v = np.reshape(
            dists_along_v - np.clip(dists_along_v, -radius, radius),
            (1, -1))
        X[Y == y, :] -= shifts_along_v.T.dot(v)

        print("Number of (%s) points projected onto slab: %s" % (y, np.sum(np.abs(dists_along_v) > radius)))

    return X


# Migrated
def get_projection_fn(X_clean, Y_clean, sphere=True, slab=True, percentile=70):
    """ Get Projection Function

    Parameters
    ----------
    X_clean : np.ndarray of shape (instances, dimensions)
        Clean input features
    Y_clean : np.ndarray of shape (instances,)
        Clean input labels
    sphere : bool
        Use sphere projection?
    slab : bool
        Use slab projection?
    percentile : float
        Percentage of data to keep when setting radii

    Returns
    -------
    project_onto_feasible_set : function
        Function to project data onto the feasible set
    """

    class_map, centroids, centroid_vec, sphere_radii, slab_radii = get_data_params(X_clean, Y_clean, percentile)
    if sphere and slab:
        projector = upper_bounds.Projector()

        def project_onto_feasible_set(X, Y):
            num_examples = X.shape[0]
            proj_X = np.zeros_like(X)
            for idx in range(num_examples):
                x = X[idx, :]
                y = Y[idx]
                class_idx = class_map[y]
                centroid = centroids[class_idx, :]
                sphere_radius = sphere_radii[class_idx]
                slab_radius = slab_radii[class_idx]
                proj_X[idx, :] = projector.project_onto_feasible_set(x, centroid, centroid_vec, sphere_radius, slab_radius)
            return proj_X

    else:
        def project_onto_feasible_set(X, Y):
            if sphere:
                X = project_onto_sphere(X, Y, sphere_radii, centroids, class_map)

            elif slab:
                X = project_onto_slab(X, Y, centroid_vec, slab_radii, centroids, class_map)
            return X

    return project_onto_feasible_set


# Migrated
def filter_points_outside_feasible_set(X, Y, centroids, centroid_vec, sphere_radii,
                                       slab_radii, class_map):
    """ Remove Points Outside of Feasible Set

    Parameters
    ----------
    X : np.ndarray of shape (instances, dimensions)
        Input features
    Y : np.ndarray of shape (instances,)
        Input labels
    centroids : np.ndarray of shape (classes, dimensions)
        Centroid for each class
    centroid_vec : np.ndarray of shape (1, dimensions)
        Vector between centroids
    sphere_radii : np.ndarray of shape (classes,)
        Radii to use with sphere projection
    slab_radii : np.ndarray of shape (classes,)
        Radii to use with slab projection
    class_map : dict
        Class to index mapping

    Returns
    -------
    X_filtered : np.ndarray of shape (instances, dimensions)
        Filtered input features
    Y_filtered : np.ndarray of shape (instances,)
        Filtered input labels
    """
    sphere_dists = defenses.compute_dists_under_Q(
        X, 
        Y, 
        Q=None, 
        centroids=centroids,
        class_map=class_map)
    slab_dists = defenses.compute_dists_under_Q(
        X, 
        Y, 
        Q=centroid_vec, 
        centroids=centroids,
        class_map=class_map)

    idx_to_keep = np.array([True] * X.shape[0])
    for y in set(Y):
        idx_to_keep[np.where(Y == y)[0][sphere_dists[Y == y] > sphere_radii[class_map[y]]]] = False
        idx_to_keep[np.where(Y == y)[0][slab_dists[Y == y] > slab_radii[class_map[y]]]] = False

    print(np.sum(idx_to_keep))
    return X[idx_to_keep, :], Y[idx_to_keep]
