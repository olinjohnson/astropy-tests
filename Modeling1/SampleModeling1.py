import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astroquery.vizier import Vizier
import scipy.optimize

catalog = Vizier.get_catalogs("J/A+A/605/A100")
period = np.array(catalog[0]["Period"])
log_period = np.log10(period)
k_mag = np.array(catalog[0]["__Ksmag_"])
k_mag_err = np.array(catalog[0]["e__Ksmag_"])

model = models.Linear1D()
fitter = fitting.LinearLSQFitter()
best_fit = fitter(model, log_period, k_mag, weights=1.0/k_mag_err)
print(best_fit)

plt.errorbar(log_period, k_mag, k_mag_err, fmt="k.")
plt.plot(log_period, best_fit(log_period), color="g", linewidth=3)
plt.xlabel(r'$\log_{10}$(Period [days])')
plt.ylabel("Ks")
plt.show()