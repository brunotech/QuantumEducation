import numpy as np
import matplotlib.pyplot as plt

class svm_utils:

	def make_meshgrid(self, y, h=.02):
		"""Create a mesh of points to plot in

	    Parameters
	    ----------
	    x: data to base x-axis meshgrid on
	    y: data to base y-axis meshgrid on
	    h: stepsize for meshgrid, optional

	    Returns
	    -------
	    xx, yy : ndarray
	    """
		x_min, x_max = self.min() - 1, self.max() + 1
		y_min, y_max = y.min() - 1, y.max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		np.arange(y_min, y_max, h))
		return xx, yy


	def plot_contours(self, clf, xx, yy, **params):
		"""Plot the decision boundaries for a classifier.

	    Parameters
	    ----------
	    ax: matplotlib axes object
	    clf: a classifier
	    xx: meshgrid ndarray
	    yy: meshgrid ndarray
	    params: dictionary of params to pass to contourf, optional
	    """
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		return self.contourf(xx, yy, Z, **params)
