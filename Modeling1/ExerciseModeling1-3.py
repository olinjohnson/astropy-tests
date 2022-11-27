import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import scipy.optimize

# Data:
N3 = 100
x3 = np.linspace(0, 3, N3)
y3 = 5.0 * np.sin(2 * np.pi * x3)
y3 = np.array([y_point + np.random.normal(0, 1) for y_point in y3])
sigma = 1.5
y3_err = np.ones(N3)*sigma 

model = models.Sine1D()
fitter = fitting.LevMarLSQFitter()
best_fit = fitter(model, x3, y3, weights=1.0/y3_err)

plt.errorbar(x3, y3, yerr=y3_err, fmt="k.")
plt.plot(x3, best_fit(x3), color="r", linewidth=3)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
