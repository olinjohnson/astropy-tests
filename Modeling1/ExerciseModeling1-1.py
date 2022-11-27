import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import scipy.optimize

# Data:
N = 100
x1 = np.linspace(0, 4, N)  # Makes an array from 0 to 4 of N elements
y1 = x1**3 - 6*x1**2 + 12*x1 - 9 
# Now we add some noise to the data
y1 += np.random.normal(0, 2, size=len(y1)) #One way to add random gaussian noise
sigma = 1.5
y1_err = np.ones(N)*sigma 

model = models.Polynomial1D(degree=3)

# fitter = fitting.SimplexLSQFitter()
fitter = fitting.LinearLSQFitter()
# Fit accuracy can be evaluated using the Reduced Chi Square Value algorithm

best_fit = fitter(model, x1, y1, weights=1.0/y1_err)

plt.errorbar(x1, y1, yerr=y1_err, fmt="k.")
plt.plot(x1, best_fit(x1), color="r", linewidth=3)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
