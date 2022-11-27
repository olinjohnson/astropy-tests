import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import scipy.optimize

# Data: (fake gaussian distribution)
N = 100
mu, sigma, amplitude = 0.0, 10.0, 10.0
N2 = 100
x2 = np.linspace(-30, 30, N)
y2 = amplitude * np.exp(-(x2-mu)**2 / (2*sigma**2))
y2 = np.array([y_point + np.random.normal(0, 1) for y_point in y2])   #Another way to add random gaussian noise
sigma = 1
y2_err = np.ones(N)*sigma

# model = models.Polynomial1D(degree=3)
# fitter = fitting.SimplexLSQFitter()
# fitter = fitting.LinearLSQFitter()
# Fit accuracy can be evaluated using the Reduced Chi Square Value algorithm
# best_fit = fitter(model, x2, y2, weights=1.0/y2_err)

gaussian_model = models.Gaussian1D()
guassian_fitter = fitting.LevMarLSQFitter()
guassian_best_fit = guassian_fitter(gaussian_model, x2, y2, weights=1.0/y2_err)

plt.errorbar(x2, y2, yerr=y2_err, fmt="k.")
plt.plot(x2, guassian_best_fit(x2), color="r", linewidth=3)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
