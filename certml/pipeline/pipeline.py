"""Pipeline"""


class Pipeline(object):
    """Pipeline"""

    def __init__(self, steps):
        self.steps = steps

    def predict(self, X):
        """Apply the transformations and predict the final estimator"""
        x_step = X
        for name, step in self.steps[:-1]:
            result = step.transform(x_step)
            try:
                x_step, used = result
            except ValueError:
                x_step = result

        return self._final_estimator.predict(x_step)

    def fit(self, X, y):
        """Apply the transformations and fit the final estimator"""
        x_step = X
        y_step = y
        for name, step in self.steps[:-1]:
            try:
                # Some transformations can be fit on data
                step.fit(x_step, y_step)
            except AttributeError:
                # The transformation does not need to be fit
                pass

            result = step.transform(x_step, y_step)
            try:
                x_step, y_step, used = result
            except ValueError:
                x_step, y_step = result

        return self._final_estimator.fit(x_step, y_step)

    def fit_trusted(self, X, y):
        """Apply the transformations and fit the final estimator"""
        x_step = X
        y_step = y
        for name, step in self.steps[:-1]:
            try:
                # Some transformations can be fit on data
                step.fit_trusted(x_step, y_step)
            except AttributeError:
                # The transformation does not need to be fit
                pass

            result = step.transform(x_step, y_step)
            try:
                x_step, y_step, used = result
            except ValueError:
                x_step, y_step = result

        try:
            # The if the final estimator has the method fit_trusted, use that.
            return self._final_estimator.fit_trusted(x_step, y_step)
        except AttributeError:
            # Otherwise, just use fit
            return self._final_estimator.fit(x_step, y_step)

    def partial_fit(self, X, y):
        """Apply the transformations and then partial fit the final estimator"""
        x_step = X
        y_step = y
        for name, step in self.steps[:-1]:
            try:
                # Some transformations can be fit on data
                step.fit(x_step, y_step)
            except AttributeError:
                # The transformation does not need to be fit
                pass

            result = step.transform(x_step, y_step)
            try:
                x_step, y_step, used = result
            except ValueError:
                x_step, y_step = result
        return self._final_estimator.partial_fit(x_step, y_step)

    def cert_params(self):
        cert_params = [None] * len(self.steps)
        for ind, (name, step) in enumerate(self.steps):
            try:
                params = step.cert_params()
                cert_params[ind] = params
            except AttributeError:
                # Not a certifiable step
                pass
        return cert_params

    @property
    def _final_estimator(self):
        return self.steps[-1][-1]
